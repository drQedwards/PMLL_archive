// pmll_persistent_concert.c
// Minimal, self-contained reference of PMLL-style persistent weight calls.
// Implements: vectorized hash anchoring, ΔW overlay, bias recall, novelty splitting.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef REAL
#define REAL float
#endif

// ---------- Utilities ----------
static uint64_t fnv1a64(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t*)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    while (len--) { h ^= *p++; h *= 0x100000001b3ULL; }
    return h;
}
static uint64_t hash_u64_combine(uint64_t a, uint64_t b) {
    // Mix two 64-bit values (splitmix64 style)
    uint64_t z = a + 0x9e3779b97f4a7c15ULL + (b<<6) + (b>>2);
    z ^= z >> 30; z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27; z *= 0x94d049bb133111ebULL;
    z ^= z >> 31; return z;
}
static uint64_t hash_vector(const REAL *v, int dim) {
    return fnv1a64(v, sizeof(REAL)*dim);
}
static double dot(const REAL *a, const REAL *b, int n){ double s=0; for(int i=0;i<n;i++) s += (double)a[i]*(double)b[i]; return s; }
static double norm(const REAL *a, int n){ return sqrt(fmax(1e-12, dot(a,a,n))); }
static double cosine(const REAL *a, const REAL *b, int n){ return dot(a,b,n)/(norm(a,n)*norm(b,n)); }
static void axpy(REAL *y, const REAL *x, REAL alpha, int n){ for(int i=0;i<n;i++) y[i] += alpha*x[i]; }
static void scal(REAL *x, REAL s, int n){ for(int i=0;i<n;i++) x[i] = (REAL)(s*x[i]); }
static void copyv(REAL *dst, const REAL *src, int n){ memcpy(dst, src, sizeof(REAL)*n); }
static uint64_t now_ms(){ struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (uint64_t)ts.tv_sec*1000ULL + ts.tv_nsec/1000000ULL; }

// ---------- Linear algebra ----------
typedef struct { int rows, cols; REAL *w; } Matrix;
typedef struct { int dim; REAL *v; } Vector;

static Matrix mat_new(int rows, int cols){
    Matrix M = {rows, cols, (REAL*)calloc((size_t)rows*cols, sizeof(REAL))};
    return M;
}
static void mat_free(Matrix *M){ free(M->w); M->w=NULL; }
static void matvec(const Matrix *W, const REAL *x, REAL *y){
    for(int r=0; r<W->rows; ++r){
        double s=0;
        const REAL *wr = &W->w[(size_t)r*W->cols];
        for(int c=0;c<W->cols;c++) s += (double)wr[c]*(double)x[c];
        y[r]=(REAL)s;
    }
}
static void matvec_accum(const Matrix *W, const REAL *x, REAL *y){
    for(int r=0; r<W->rows; ++r){
        double s=0;
        const REAL *wr = &W->w[(size_t)r*W->cols];
        for(int c=0;c<W->cols;c++) s += (double)wr[c]*(double)x[c];
        y[r]+=(REAL)s;
    }
}
static void outer_accum(Matrix *A, const REAL *u, const REAL *v, REAL alpha){
    for(int r=0;r<A->rows;r++){
        REAL *ar = &A->w[(size_t)r*A->cols];
        for(int c=0;c<A->cols;c++){
            ar[c] += alpha * u[r] * v[c];
        }
    }
}
static void mat_scal(Matrix *A, REAL s){
    for(int i=0;i<A->rows*A->cols;i++) A->w[i]*=s;
}

// ---------- Lattice + traces ----------
typedef struct {
    uint64_t t_ms;
    uint64_t prev_anchor;
    uint64_t anchor;
    uint64_t hx, hy, hctx;
    double   novelty;           // heuristic novelty score at write time
    // store small snapshots to enable bias recall
    int y_dim, ctx_dim;
    REAL *y_snapshot;           // output snapshot at that time
    REAL *ctx_snapshot;         // context embedding snapshot
} Trace;

typedef struct {
    int from, to;
    float weight;
} Edge;

typedef struct {
    int id;
    Matrix W_base;      // frozen (original) weights
    Matrix dW_overlay;  // learned overlay (persistent, graph-driven)
    Vector bias_accum;  // running bias cache (for demo)
    // traces (append-only chain)
    Trace *tr; int tr_n, tr_cap;
    // adjacency edges
    Edge  *edges; int e_n, e_cap;
    uint64_t last_anchor;
} Node;

typedef struct {
    Node *nodes; int n_nodes;
} Lattice;

// ---------- Node / lattice helpers ----------
static void node_init(Node *N, int id, int in_dim, int out_dim){
    N->id = id;
    N->W_base = mat_new(out_dim, in_dim);
    N->dW_overlay = mat_new(out_dim, in_dim);
    N->bias_accum.dim = out_dim;
    N->bias_accum.v = (REAL*)calloc(out_dim, sizeof(REAL));
    N->tr=NULL; N->tr_n=0; N->tr_cap=0;
    N->edges=NULL; N->e_n=0; N->e_cap=0;
    N->last_anchor = 0;
}
static void node_free(Node *N){
    mat_free(&N->W_base);
    mat_free(&N->dW_overlay);
    free(N->bias_accum.v);
    for(int i=0;i<N->tr_n;i++){ free(N->tr[i].y_snapshot); free(N->tr[i].ctx_snapshot); }
    free(N->tr);
    free(N->edges);
}
static void push_edge(Node *N, int to, float w){
    if(N->e_n==N->e_cap){ N->e_cap = N->e_cap? N->e_cap*2:8; N->edges=(Edge*)realloc(N->edges, sizeof(Edge)*N->e_cap); }
    N->edges[N->e_n++] = (Edge){ .from = N->id, .to = to, .weight = w };
}
static void push_trace(Node *N, const Trace *T){
    if(N->tr_n==N->tr_cap){ N->tr_cap = N->tr_cap? N->tr_cap*2:16; N->tr=(Trace*)realloc(N->tr, sizeof(Trace)*N->tr_cap); }
    N->tr[N->tr_n++] = *T;
}

// ---------- Heuristics ----------
typedef struct {
    // recall weights
    double recency_half_life_ms;   // e.g., 6 hours -> 21600000
    double sim_floor;              // min cosine to consider
    double novelty_thresh;         // trigger new memory line
    // mixing
    double alpha_bias;             // strength of bias term
    double beta_overlay;           // strength of ΔW overlay
    double gamma_overlay_update;   // EMA for overlay update
} Heur;

static double recency_weight(uint64_t now_ms_ts, uint64_t then_ms, double half_life_ms){
    if(half_life_ms<=1.0) return 1.0;
    double dt = (double)(now_ms_ts - then_ms);
    return pow(0.5, dt/half_life_ms);
}

static void recall_bias_from_traces(const Node *N, const REAL *ctx, int ctx_dim,
                                    REAL *bias_out /*len=out_dim*/, const Heur *H)
{
    memset(bias_out, 0, sizeof(REAL)*N->bias_accum.dim);
    if(N->tr_n==0) return;
    uint64_t tnow = now_ms();
    double acc_w = 0.0;

    for(int i=N->tr_n-1;i>=0;i--){ // backward favors recency
        const Trace *T = &N->tr[i];
        if(T->ctx_dim!=ctx_dim || T->y_dim!=N->bias_accum.dim) continue;
        double sim = cosine(ctx, T->ctx_snapshot, ctx_dim);
        if(sim < H->sim_floor) continue;
        double rw = recency_weight(tnow, T->t_ms, H->recency_half_life_ms);
        double w  = sim * rw;
        for(int k=0;k<N->bias_accum.dim;k++) bias_out[k] += (REAL)(w * T->y_snapshot[k]);
        acc_w += w;
        // small early-exit if enough mass accumulated
        if(acc_w > 1.5) break;
    }
    if(acc_w > 1e-9) scal(bias_out, (REAL)(1.0/acc_w), N->bias_accum.dim);
}

static double novelty_score_against_traces(const Node *N, const REAL *ctx, int ctx_dim){
    if(N->tr_n==0) return 1.0; // first write => "max novelty"
    double best = -1.0;
    for(int i=N->tr_n-1;i>=0 && i>=N->tr_n-64;i--){ // limit window
        const Trace *T = &N->tr[i];
        if(T->ctx_dim!=ctx_dim) continue;
        double sim = cosine(ctx, T->ctx_snapshot, ctx_dim);
        if(sim>best) best=sim;
    }
    if(best<0) best=0;
    return 1.0 - best; // higher = more novel
}

// ---------- Persistent call ----------
// Implements: y = (W_base + beta*ΔW) x  +  alpha * recall_bias(ctx)
// Also writes a sealed trace and, if novel, starts a new memory line (edge).

typedef struct {
    Vector y_out;
    uint64_t anchor;
    double novelty;
} CallResult;

static CallResult persistent_weight_call(Node *N,
                                         const REAL *x, int in_dim,
                                         const REAL *ctx, int ctx_dim,
                                         const Heur *H)
{
    CallResult R = {0};
    R.y_out.dim = N->bias_accum.dim;
    R.y_out.v   = (REAL*)calloc(N->bias_accum.dim, sizeof(REAL));

    // 1) Effective weights: W_eff = W_base + beta * dW_overlay
    // compute y = W_base * x
    matvec(&N->W_base, x, R.y_out.v);
    // y += beta * (dW_overlay * x)
    REAL *tmp = (REAL*)calloc(N->bias_accum.dim, sizeof(REAL));
    matvec(&N->dW_overlay, x, tmp);
    axpy(R.y_out.v, tmp, (REAL)H->beta_overlay, N->bias_accum.dim);

    // 2) Recall bias from lattice
    REAL *bias = (REAL*)calloc(N->bias_accum.dim, sizeof(REAL));
    recall_bias_from_traces(N, ctx, ctx_dim, bias, H);
    axpy(R.y_out.v, bias, (REAL)H->alpha_bias, N->bias_accum.dim);

    // 3) Novelty check to decide line split / adjacency
    R.novelty = novelty_score_against_traces(N, ctx, ctx_dim);
    int split_line = (R.novelty >= H->novelty_thresh);

    // 4) Update ΔW overlay (EMA) with a low-rank hint from current step
    //    Use outer( normalized(y), normalized(x) ) as a tiny stabilizer.
    //    Weight it by (novelty + small recency term).
    REAL *y_norm = (REAL*)malloc(sizeof(REAL)*N->bias_accum.dim);
    REAL *x_norm = (REAL*)malloc(sizeof(REAL)*in_dim);
    copyv(y_norm, R.y_out.v, N->bias_accum.dim);
    copyv(x_norm, x, in_dim);
    scal(y_norm, (REAL)(1.0/norm(y_norm, N->bias_accum.dim)), N->bias_accum.dim);
    scal(x_norm, (REAL)(1.0/norm(x_norm, in_dim)), in_dim);

    // decay + accumulate
    mat_scal(&N->dW_overlay, (REAL)(1.0 - H->gamma_overlay_update));
    outer_accum(&N->dW_overlay, y_norm, x_norm, (REAL)(H->gamma_overlay_update * (0.5 + 0.5*R.novelty)));

    free(y_norm); free(x_norm);

    // 5) Seal + write trace (anchored hash chain)
    Trace T = {0};
    T.t_ms = now_ms();
    T.prev_anchor = N->last_anchor;

    uint64_t hx   = hash_vector(x, in_dim);
    uint64_t hy   = hash_vector(R.y_out.v, N->bias_accum.dim);
    uint64_t hctx = hash_vector(ctx, ctx_dim);

    T.hx = hx; T.hy = hy; T.hctx = hctx;
    T.novelty = R.novelty;
    T.y_dim = N->bias_accum.dim; T.ctx_dim = ctx_dim;
    T.y_snapshot   = (REAL*)malloc(sizeof(REAL)*T.y_dim);
    T.ctx_snapshot = (REAL*)malloc(sizeof(REAL)*T.ctx_dim);
    copyv(T.y_snapshot, R.y_out.v, T.y_dim);
    copyv(T.ctx_snapshot, ctx, T.ctx_dim);

    // anchor = H(prev_anchor || hx || hy || hctx)
    uint64_t a = hash_u64_combine(N->last_anchor, hx);
    a = hash_u64_combine(a, hy);
    a = hash_u64_combine(a, hctx);
    T.anchor = a;
    N->last_anchor = a;

    push_trace(N, &T);

    // 6) If novelty split, add an adjacency edge (new memory line neighbor)
    if(split_line){
        // Self-adjacency represents "new line branching from current node"
        push_edge(N, N->id /*adjacent neighbor*/, (float)(0.5 + 0.5*R.novelty));
    }

    // 7) Small running bias cache (not required, useful in practice)
    axpy(N->bias_accum.v, R.y_out.v, 0.01f, N->bias_accum.dim);

    R.anchor = a;
    free(tmp); free(bias);
    return R;
}

// ---------- Demo ----------
static void seed_matrix(Matrix *M, unsigned seed){
    // deterministic tiny init
    uint32_t s = seed;
    for(int i=0;i<M->rows*M->cols;i++){
        s = 1664525u*s + 1013904223u;
        M->w[i] = (REAL)((int)(s>>9)%1000 - 500)/5000.0f; // ~[-0.1,0.1]
    }
}
static void print_vec(const char *tag, const REAL *v, int n){
    printf("%s[", tag);
    for(int i=0;i<n;i++){ printf("%s%.4f", (i?", ":""), (double)v[i]); }
    printf("]\n");
}

int main(void){
    const int IN=8, OUT=6, CTX=8;

    Node N; node_init(&N, /*id*/1, IN, OUT);
    seed_matrix(&N.W_base, 42u);

    Heur H = {
        .recency_half_life_ms = 6.0*60*60*1000.0, // 6h
        .sim_floor = 0.15,
        .novelty_thresh = 0.45,
        .alpha_bias = 0.35,
        .beta_overlay = 0.50,
        .gamma_overlay_update = 0.08
    };

    REAL x[IN]   = {0.9f,0.2f,0.1f,0.0f, -0.4f,0.3f,0.0f,0.2f};
    REAL ctx[CTX]= {0.1f,0.0f,0.7f,0.0f,  0.2f,0.1f,0.3f,0.5f};

    // Simulate several calls across evolving contexts
    for(int step=0; step<5; ++step){
        // drift context slightly (novel + adjacent)
        for(int i=0;i<CTX;i++){ ctx[i] += (REAL)((i%3==0)? 0.03f : -0.01f); }
        CallResult R = persistent_weight_call(&N, x, IN, ctx, CTX, &H);

        printf("step %d: anchor=%016llx novelty=%.3f\n",
               step, (unsigned long long)R.anchor, R.novelty);
        print_vec("y*", R.y_out.v, R.y_out.dim);
        free(R.y_out.v);

        // small input drift too
        for(int i=0;i<IN;i++){ x[i] += (REAL)((i%2)? 0.02f : -0.01f); }
    }

    // Inspect chain + edges
    printf("\nTraces: %d | Edges: %d\n", N.tr_n, N.e_n);
    if(N.tr_n>0){
        printf("last anchor: %016llx\n", (unsigned long long)N.last_anchor);
    }
    for(int i=0;i<N.e_n;i++){
        printf("edge %d: %d -> %d (w=%.3f)\n", i, N.edges[i].from, N.edges[i].to, N.edges[i].weight);
    }

    node_free(&N);
    return 0;
}
