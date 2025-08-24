// pmll_claude_bridge.c
// Persistent Memory Lattice for bridging Claude conversation contexts
// Extends PMLL to support cross-conversation state persistence

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mmap.h>
#include <fcntl.h>

#ifndef REAL
#define REAL float
#endif

#define EMBEDDING_DIM 768  // Typical transformer embedding size
#define MAX_CONV_TRACES 10000
#define MEMORY_POOL_SIZE (1024*1024*100) // 100MB persistent pool

// ---------- Core utilities (inherited from original) ----------
static uint64_t fnv1a64(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t*)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    while (len--) { h ^= *p++; h *= 0x100000001b3ULL; }
    return h;
}

static uint64_t hash_u64_combine(uint64_t a, uint64_t b) {
    uint64_t z = a + 0x9e3779b97f4a7c15ULL + (b<<6) + (b>>2);
    z ^= z >> 30; z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27; z *= 0x94d049bb133111ebULL;
    z ^= z >> 31; return z;
}

static double cosine(const REAL *a, const REAL *b, int n) {
    double dot_prod = 0, norm_a = 0, norm_b = 0;
    for(int i = 0; i < n; i++) {
        dot_prod += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot_prod / (sqrt(norm_a) * sqrt(norm_b) + 1e-12);
}

static uint64_t now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec*1000ULL + ts.tv_nsec/1000000ULL;
}

// ---------- Extended structures for Claude bridging ----------

typedef enum {
    CONV_STATE_ACTIVE = 1,
    CONV_STATE_DORMANT = 2,
    CONV_STATE_ARCHIVED = 3
} ConvState;

typedef enum {
    MEM_TYPE_QUERY = 1,
    MEM_TYPE_RESPONSE = 2,
    MEM_TYPE_CONTEXT = 3,
    MEM_TYPE_REFLECTION = 4  // Meta-cognitive traces
} MemoryType;

// Conversation-level trace with embeddings
typedef struct {
    // Identity
    uint64_t conversation_id;
    uint64_t message_seq;
    uint64_t timestamp_ms;
    
    // Chain anchoring (across conversations)
    uint64_t local_anchor;      // within conversation
    uint64_t global_anchor;     // across all conversations
    uint64_t prev_conv_anchor;  // link to previous conversation's last anchor
    
    // Embeddings
    REAL query_emb[EMBEDDING_DIM];
    REAL response_emb[EMBEDDING_DIM];
    REAL context_emb[EMBEDDING_DIM];
    
    // Metadata
    MemoryType type;
    float importance_score;
    float novelty_score;
    float emotional_valence;  // -1 to 1, for emotional context
    
    // Semantic tags (hashed for privacy)
    uint64_t topic_hashes[5];
    int n_topics;
    
    // Retrieval optimization
    uint64_t bloom_filter;  // Fast membership testing
    
} ConversationTrace;

// Cross-conversation edge
typedef struct {
    uint64_t from_conv_id;
    uint64_t to_conv_id;
    uint64_t from_anchor;
    uint64_t to_anchor;
    float weight;
    float semantic_similarity;
    int hop_distance;  // For multi-hop retrieval
} CrossConvEdge;

// Memory index for fast retrieval
typedef struct {
    ConversationTrace *traces;
    int n_traces;
    int capacity;
    
    CrossConvEdge *edges;
    int n_edges;
    int edge_capacity;
    
    // Indices for fast lookup
    uint64_t *conv_id_index;  // Sorted for binary search
    uint64_t *anchor_index;
    
    // Memory-mapped persistence
    void *mmap_base;
    size_t mmap_size;
    int persist_fd;
    
    pthread_rwlock_t lock;
} MemoryIndex;

// Claude context bridge
typedef struct {
    // Current conversation state
    uint64_t current_conv_id;
    uint64_t current_msg_seq;
    ConvState state;
    
    // Embedding extraction interface
    int (*extract_embeddings)(const char *text, REAL *embedding, int dim);
    
    // Persistence layer
    MemoryIndex *mem_index;
    
    // Working memory (active context)
    ConversationTrace *working_memory;
    int working_memory_size;
    int working_memory_capacity;
    
    // Recall parameters
    struct {
        double recency_weight;
        double similarity_threshold;
        double novelty_bonus;
        int max_recall_items;
        int max_hop_distance;
    } recall_params;
    
} ClaudeBridge;

// ---------- Memory persistence layer ----------

static MemoryIndex* memory_index_create(const char *persist_path) {
    MemoryIndex *idx = (MemoryIndex*)calloc(1, sizeof(MemoryIndex));
    
    // Initialize memory-mapped file for persistence
    if (persist_path) {
        idx->persist_fd = open(persist_path, O_RDWR | O_CREAT, 0644);
        if (idx->persist_fd >= 0) {
            // Extend file to desired size
            if (ftruncate(idx->persist_fd, MEMORY_POOL_SIZE) == 0) {
                idx->mmap_base = mmap(NULL, MEMORY_POOL_SIZE, 
                                     PROT_READ | PROT_WRITE, 
                                     MAP_SHARED, idx->persist_fd, 0);
                idx->mmap_size = MEMORY_POOL_SIZE;
                
                // Map structures into persistent memory
                idx->traces = (ConversationTrace*)idx->mmap_base;
                idx->capacity = MEMORY_POOL_SIZE / sizeof(ConversationTrace) / 2;
                idx->edges = (CrossConvEdge*)((char*)idx->mmap_base + 
                            idx->capacity * sizeof(ConversationTrace));
                idx->edge_capacity = idx->capacity;
            }
        }
    }
    
    // Fallback to heap allocation
    if (!idx->mmap_base) {
        idx->capacity = MAX_CONV_TRACES;
        idx->traces = (ConversationTrace*)calloc(idx->capacity, sizeof(ConversationTrace));
        idx->edge_capacity = idx->capacity * 2;
        idx->edges = (CrossConvEdge*)calloc(idx->edge_capacity, sizeof(CrossConvEdge));
    }
    
    pthread_rwlock_init(&idx->lock, NULL);
    return idx;
}

static void memory_index_free(MemoryIndex *idx) {
    if (idx->mmap_base) {
        munmap(idx->mmap_base, idx->mmap_size);
        close(idx->persist_fd);
    } else {
        free(idx->traces);
        free(idx->edges);
    }
    free(idx->conv_id_index);
    free(idx->anchor_index);
    pthread_rwlock_destroy(&idx->lock);
    free(idx);
}

// ---------- Embedding extraction (placeholder for real implementation) ----------

static int extract_embeddings_simulated(const char *text, REAL *embedding, int dim) {
    // In production, this would call a real embedding model
    // For now, simulate with hash-based pseudo-embeddings
    uint64_t h = fnv1a64(text, strlen(text));
    
    for (int i = 0; i < dim; i++) {
        h = hash_u64_combine(h, i);
        embedding[i] = (REAL)((h % 1000) / 500.0f - 1.0f);
    }
    
    // Normalize
    float norm = 0;
    for (int i = 0; i < dim; i++) norm += embedding[i] * embedding[i];
    norm = sqrt(norm);
    for (int i = 0; i < dim; i++) embedding[i] /= norm;
    
    return 0;
}

// ---------- Core bridge operations ----------

static ClaudeBridge* bridge_create(const char *persist_path) {
    ClaudeBridge *bridge = (ClaudeBridge*)calloc(1, sizeof(ClaudeBridge));
    
    bridge->mem_index = memory_index_create(persist_path);
    bridge->extract_embeddings = extract_embeddings_simulated;
    
    // Default recall parameters
    bridge->recall_params.recency_weight = 0.3;
    bridge->recall_params.similarity_threshold = 0.65;
    bridge->recall_params.novelty_bonus = 0.2;
    bridge->recall_params.max_recall_items = 10;
    bridge->recall_params.max_hop_distance = 3;
    
    bridge->working_memory_capacity = 100;
    bridge->working_memory = (ConversationTrace*)calloc(
        bridge->working_memory_capacity, sizeof(ConversationTrace));
    
    return bridge;
}

static void bridge_free(ClaudeBridge *bridge) {
    memory_index_free(bridge->mem_index);
    free(bridge->working_memory);
    free(bridge);
}

// ---------- Write new trace ----------

static uint64_t bridge_write_trace(ClaudeBridge *bridge,
                                   const char *query_text,
                                   const char *response_text,
                                   const char *context_summary) {
    pthread_rwlock_wrlock(&bridge->mem_index->lock);
    
    MemoryIndex *idx = bridge->mem_index;
    if (idx->n_traces >= idx->capacity - 1) {
        // In production: implement compaction/archival
        pthread_rwlock_unlock(&idx->lock);
        return 0;
    }
    
    ConversationTrace *trace = &idx->traces[idx->n_traces];
    
    // Basic fields
    trace->conversation_id = bridge->current_conv_id;
    trace->message_seq = bridge->current_msg_seq++;
    trace->timestamp_ms = now_ms();
    
    // Extract embeddings
    bridge->extract_embeddings(query_text, trace->query_emb, EMBEDDING_DIM);
    bridge->extract_embeddings(response_text, trace->response_emb, EMBEDDING_DIM);
    bridge->extract_embeddings(context_summary, trace->context_emb, EMBEDDING_DIM);
    
    // Compute anchors (chained hashing)
    uint64_t prev_anchor = (idx->n_traces > 0) ? 
                          idx->traces[idx->n_traces-1].local_anchor : 0;
    
    trace->local_anchor = hash_u64_combine(prev_anchor, 
                         hash_u64_combine(fnv1a64(query_text, strlen(query_text)),
                                        fnv1a64(response_text, strlen(response_text))));
    
    // Global anchor chains across conversations
    trace->global_anchor = hash_u64_combine(trace->conversation_id, trace->local_anchor);
    
    if (idx->n_traces > 0) {
        // Find previous conversation's last trace
        for (int i = idx->n_traces - 1; i >= 0; i--) {
            if (idx->traces[i].conversation_id != bridge->current_conv_id) {
                trace->prev_conv_anchor = idx->traces[i].global_anchor;
                break;
            }
        }
    }
    
    // Compute importance and novelty
    trace->importance_score = 0.5f;  // Default, would be computed from engagement
    
    // Novelty: how different from recent traces?
    float max_similarity = 0;
    int check_window = (idx->n_traces < 50) ? idx->n_traces : 50;
    for (int i = idx->n_traces - check_window; i < idx->n_traces; i++) {
        if (i < 0) continue;
        float sim = cosine(trace->context_emb, idx->traces[i].context_emb, EMBEDDING_DIM);
        if (sim > max_similarity) max_similarity = sim;
    }
    trace->novelty_score = 1.0f - max_similarity;
    
    // Add to working memory
    if (bridge->working_memory_size < bridge->working_memory_capacity) {
        bridge->working_memory[bridge->working_memory_size++] = *trace;
    }
    
    idx->n_traces++;
    
    // Check for cross-conversation edges
    if (trace->novelty_score < 0.7f) {  // Similar to something
        // Find similar traces in other conversations
        for (int i = 0; i < idx->n_traces - 1; i++) {
            if (idx->traces[i].conversation_id == bridge->current_conv_id) continue;
            
            float sim = cosine(trace->context_emb, idx->traces[i].context_emb, EMBEDDING_DIM);
            if (sim > bridge->recall_params.similarity_threshold) {
                // Create cross-conversation edge
                if (idx->n_edges < idx->edge_capacity) {
                    CrossConvEdge *edge = &idx->edges[idx->n_edges++];
                    edge->from_conv_id = bridge->current_conv_id;
                    edge->to_conv_id = idx->traces[i].conversation_id;
                    edge->from_anchor = trace->global_anchor;
                    edge->to_anchor = idx->traces[i].global_anchor;
                    edge->semantic_similarity = sim;
                    edge->weight = sim * trace->importance_score;
                    edge->hop_distance = 1;
                }
            }
        }
    }
    
    pthread_rwlock_unlock(&idx->lock);
    return trace->global_anchor;
}

// ---------- Multi-hop memory retrieval ----------

typedef struct {
    ConversationTrace *trace;
    float score;
    int hop_distance;
} RecallCandidate;

static int recall_compare(const void *a, const void *b) {
    float score_a = ((RecallCandidate*)a)->score;
    float score_b = ((RecallCandidate*)b)->score;
    return (score_a < score_b) - (score_a > score_b);
}

static int bridge_recall_relevant(ClaudeBridge *bridge,
                                  const char *query_context,
                                  ConversationTrace **out_traces,
                                  int max_items) {
    pthread_rwlock_rdlock(&bridge->mem_index->lock);
    
    REAL query_emb[EMBEDDING_DIM];
    bridge->extract_embeddings(query_context, query_emb, EMBEDDING_DIM);
    
    MemoryIndex *idx = bridge->mem_index;
    RecallCandidate *candidates = (RecallCandidate*)calloc(idx->n_traces, sizeof(RecallCandidate));
    int n_candidates = 0;
    
    uint64_t now = now_ms();
    
    // Direct retrieval (hop distance = 0)
    for (int i = 0; i < idx->n_traces; i++) {
        ConversationTrace *t = &idx->traces[i];
        
        // Skip current conversation's traces (we already have those in context)
        if (t->conversation_id == bridge->current_conv_id) continue;
        
        float sim = cosine(query_emb, t->context_emb, EMBEDDING_DIM);
        if (sim < bridge->recall_params.similarity_threshold) continue;
        
        // Compute retrieval score
        float recency = exp(-(now - t->timestamp_ms) / (7.0 * 24 * 60 * 60 * 1000.0)); // 7-day half-life
        float score = sim * (1.0 + bridge->recall_params.novelty_bonus * t->novelty_score) *
                     (1.0 + bridge->recall_params.recency_weight * recency) *
                     t->importance_score;
        
        candidates[n_candidates].trace = t;
        candidates[n_candidates].score = score;
        candidates[n_candidates].hop_distance = 0;
        n_candidates++;
    }
    
    // Multi-hop retrieval via edges
    if (bridge->recall_params.max_hop_distance > 0) {
        for (int hop = 1; hop <= bridge->recall_params.max_hop_distance; hop++) {
            int prev_n = n_candidates;
            
            for (int i = 0; i < idx->n_edges; i++) {
                CrossConvEdge *e = &idx->edges[i];
                
                // Check if edge connects to any current candidate
                for (int j = 0; j < prev_n; j++) {
                    if (candidates[j].hop_distance != hop - 1) continue;
                    
                    if (candidates[j].trace->global_anchor == e->from_anchor ||
                        candidates[j].trace->global_anchor == e->to_anchor) {
                        
                        // Find the other end of the edge
                        uint64_t target_anchor = (candidates[j].trace->global_anchor == e->from_anchor) ?
                                               e->to_anchor : e->from_anchor;
                        
                        // Find trace with target anchor
                        for (int k = 0; k < idx->n_traces; k++) {
                            if (idx->traces[k].global_anchor == target_anchor) {
                                // Decay score by hop distance
                                float hop_decay = pow(0.7, hop);
                                float score = candidates[j].score * e->weight * hop_decay;
                                
                                // Check if already in candidates
                                int exists = 0;
                                for (int m = 0; m < n_candidates; m++) {
                                    if (candidates[m].trace == &idx->traces[k]) {
                                        if (score > candidates[m].score) {
                                            candidates[m].score = score;
                                            candidates[m].hop_distance = hop;
                                        }
                                        exists = 1;
                                        break;
                                    }
                                }
                                
                                if (!exists && n_candidates < idx->n_traces) {
                                    candidates[n_candidates].trace = &idx->traces[k];
                                    candidates[n_candidates].score = score;
                                    candidates[n_candidates].hop_distance = hop;
                                    n_candidates++;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Sort by score and return top items
    qsort(candidates, n_candidates, sizeof(RecallCandidate), recall_compare);
    
    int n_return = (n_candidates < max_items) ? n_candidates : max_items;
    for (int i = 0; i < n_return; i++) {
        out_traces[i] = candidates[i].trace;
    }
    
    free(candidates);
    pthread_rwlock_unlock(&idx->lock);
    
    return n_return;
}

// ---------- Context injection formatter ----------

static char* format_recalled_context(ConversationTrace **traces, int n_traces) {
    // Format recalled memories for injection into Claude's context
    size_t buf_size = 4096 + n_traces * 512;
    char *buffer = (char*)malloc(buf_size);
    int offset = 0;
    
    offset += snprintf(buffer + offset, buf_size - offset,
                      "# Retrieved Context from Previous Conversations\n\n");
    
    for (int i = 0; i < n_traces; i++) {
        ConversationTrace *t = traces[i];
        
        // Convert timestamp to human-readable
        time_t ts_sec = t->timestamp_ms / 1000;
        struct tm *tm_info = localtime(&ts_sec);
        char time_str[64];
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M", tm_info);
        
        offset += snprintf(buffer + offset, buf_size - offset,
                          "## Memory %d (Conv: %llx, Time: %s, Relevance: %.1f%%)\n",
                          i + 1, 
                          (unsigned long long)t->conversation_id,
                          time_str,
                          t->importance_score * 100);
        
        // Add semantic indicators
        offset += snprintf(buffer + offset, buf_size - offset,
                          "- Novelty: %.2f | Importance: %.2f | Type: %s\n",
                          t->novelty_score,
                          t->importance_score,
                          (t->type == MEM_TYPE_QUERY) ? "Query" :
                          (t->type == MEM_TYPE_RESPONSE) ? "Response" :
                          (t->type == MEM_TYPE_CONTEXT) ? "Context" : "Reflection");
        
        // Topic hashes as abstract identifiers
        if (t->n_topics > 0) {
            offset += snprintf(buffer + offset, buf_size - offset, "- Topics: ");
            for (int j = 0; j < t->n_topics; j++) {
                offset += snprintf(buffer + offset, buf_size - offset, 
                                 "%llx ", (unsigned long long)t->topic_hashes[j]);
            }
            offset += snprintf(buffer + offset, buf_size - offset, "\n");
        }
        
        offset += snprintf(buffer + offset, buf_size - offset,
                          "- Anchor: %llx → %llx\n\n",
                          (unsigned long long)t->prev_conv_anchor,
                          (unsigned long long)t->global_anchor);
    }
    
    return buffer;
}

// ---------- Main API for Claude integration ----------

typedef struct {
    ClaudeBridge *bridge;
    char *injected_context;
    ConversationTrace *recalled_traces[100];
    int n_recalled;
} ClaudeMemoryAPI;

static ClaudeMemoryAPI* claude_memory_init(const char *persist_path) {
    ClaudeMemoryAPI *api = (ClaudeMemoryAPI*)calloc(1, sizeof(ClaudeMemoryAPI));
    api->bridge = bridge_create(persist_path);
    return api;
}

static void claude_memory_free(ClaudeMemoryAPI *api) {
    bridge_free(api->bridge);
    free(api->injected_context);
    free(api);
}

static void claude_memory_start_conversation(ClaudeMemoryAPI *api, uint64_t conv_id) {
    api->bridge->current_conv_id = conv_id;
    api->bridge->current_msg_seq = 0;
    api->bridge->state = CONV_STATE_ACTIVE;
    api->bridge->working_memory_size = 0;
}

static char* claude_memory_prepare_context(ClaudeMemoryAPI *api, const char *user_query) {
    // Recall relevant memories
    api->n_recalled = bridge_recall_relevant(api->bridge, user_query, 
                                            api->recalled_traces, 
                                            api->bridge->recall_params.max_recall_items);
    
    // Format for injection
    free(api->injected_context);
    api->injected_context = format_recalled_context(api->recalled_traces, api->n_recalled);
    
    return api->injected_context;
}

static uint64_t claude_memory_record_exchange(ClaudeMemoryAPI *api,
                                             const char *user_query,
                                             const char *claude_response,
                                             const char *context_summary) {
    return bridge_write_trace(api->bridge, user_query, claude_response, context_summary);
}

// ---------- Demo/Test ----------

static void demo_cross_conversation_memory() {
    printf("=== PMLL Claude Bridge Demo ===\n\n");
    
    ClaudeMemoryAPI *api = claude_memory_init("./claude_memory.dat");
    
    // Simulate first conversation
    printf("Starting Conversation 1...\n");
    claude_memory_start_conversation(api, 0x1001);
    
    const char *conv1_queries[] = {
        "Tell me about quantum computing basics",
        "How do qubits differ from classical bits?",
        "What is quantum entanglement?"
    };
    
    const char *conv1_responses[] = {
        "Quantum computing uses quantum mechanics principles...",
        "Qubits can exist in superposition states...",
        "Entanglement creates correlated quantum states..."
    };
    
    for (int i = 0; i < 3; i++) {
        uint64_t anchor = claude_memory_record_exchange(api,
            conv1_queries[i], conv1_responses[i], "quantum physics discussion");
        printf("  Message %d → Anchor: %016llx\n", i+1, (unsigned long long)anchor);
    }
    
    // Simulate second conversation (different topic)
    printf("\nStarting Conversation 2...\n");
    claude_memory_start_conversation(api, 0x1002);
    
    const char *conv2_queries[] = {
        "Explain machine learning basics",
        "What is backpropagation?",
        "How do neural networks learn?"
    };
    
    const char *conv2_responses[] = {
        "Machine learning enables pattern recognition...",
        "Backpropagation computes gradients...",
        "Neural networks adjust weights through training..."
    };
    
    for (int i = 0; i < 3; i++) {
        uint64_t anchor = claude_memory_record_exchange(api,
            conv2_queries[i], conv2_responses[i], "ML/AI discussion");
        printf("  Message %d → Anchor: %016llx\n", i+1, (unsigned long long)anchor);
    }
    
    // Third conversation - bridge to quantum ML
    printf("\nStarting Conversation 3 (Bridging topics)...\n");
    claude_memory_start_conversation(api, 0x1003);
    
    const char *bridge_query = "How might quantum computing enhance machine learning?";
    
    // This should recall from BOTH previous conversations
    char *context = claude_memory_prepare_context(api, bridge_query);
    printf("\nRecalled %d relevant memories:\n%s\n", api->n_recalled, context);
    
    // Record the bridging response
    uint64_t bridge_anchor = claude_memory_record_exchange(api,
        bridge_query,
        "Quantum computing could accelerate ML through quantum algorithms...",
        "quantum-ML intersection");
    
    printf("Bridge anchor: %016llx\n", (unsigned long long)bridge_anchor);
    
    // Show memory graph stats
    printf("\n=== Memory Statistics ===\n");
    printf("Total traces: %d\n", api->bridge->mem_index->n_traces);
    printf("Cross-conversation edges: %d\n", api->bridge->mem_index->n_edges);
    printf("Working memory size: %d\n", api->bridge->working_memory_size);
    
    claude_memory_free(api);
}

int main(void) {
    demo_cross_conversation_memory();
    return 0;
}
