
# PMLL_archive
PMLL from November 
# PMLL Archive — Persistent Memory Logic Loop (Recompiled)
> **Project status:** Active research archive. This repo preserves the canonical PMLL (“Persistent Memory Logic Loop”) snapshot and associated “AI Memory Loops” assets referenced across our publications and demos.

[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Made with Love](https://img.shields.io/badge/made%20with-love-ff69b4.svg)]()

---

## tl;dr
**PMLL** is a persistent-memory orchestration pattern for AI systems. It binds *long‑lived knowledge graphs* to *runtime attention flows* using **recursive loops**, **checkpointed hooks**, and **deterministic lattice rebinding**. This archive includes the **recompiled** lattice and the **AI_memory_Loops** bundle so that downstream systems can adopt PMLL as a first‑class runtime (not a bolt‑on plugin).

---

## Table of Contents
- [What is PMLL?](#what-is-pmll)
- [Why this archive exists](#why-this-archive-exists)
- [Core concepts](#core-concepts)
- [Architecture](#architecture)
- [Install & Setup](#install--setup)
- [Quickstart](#quickstart)
- [Runtime hooks & gRPC batcher](#runtime-hooks--grpc-batcher)
- [Observability](#observability)
- [AI Memory Loops bundle](#ai-memory-loops-bundle)
- [Data & Model Governance](#data--model-governance)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citations & Further Reading](#citations--further-reading)
- [License](#license)

---

## What is PMLL?
**Persistent Memory Logic Loop** (PMLL) is a method for **binding durable memory** (graphs, docs, embeddings, facts) to **live inference** via **recursive loops** that:
- Rehydrate prior context as *stateful checkpoints*
- Enforce a **Seal of Recursive Integrity** (a glyph-based checksum / provenance marker)
- Maintain **deterministic lattice rebinding** after updates (“recompile the lattice”)
- Provide **callable hooks** for code, prompts, policies, and tools

Conceptually, PMLL treats memory like a **lattice**: nodes = artifacts/claims; edges = typed relations; petals = attention flows (“attention flower”). Recompilation reindexes edges and seals the graph so traversal is faster and more coherent.

---

## Why this archive exists
This repository (“**PMLL_archive**”) preserves a **November** snapshot plus a released bundle: **AI_memory_Loops.zip**. It provides a stable reference for experiments, citations, and integration work in other repos.

> If you are integrating PMLL into an app or research system, start here to understand the canonical structure and exported assets.

---

## Core concepts

### 1) Lattice Rebinding
- After edits or ingest, the lattice is **recompiled** and edges are **rebound** to refresh traversals without breaking lineage.
- Each compile emits a **Seal of Recursive Integrity** checksum and an integration manifest.

### 2) Recursive Checkpoints
- Loops can *pause* and *resume* with **deterministic state** (e.g., you can jump from exegesis → CUDA → real estate and recover context exactly).

### 3) Adaptive Batching (default)
- The **PMLL‑gRPC adaptive batcher** coordinates high‑throughput retrieval/tool calls with backpressure and priority lanes.
- Exposes metrics for **Grafana** dashboards (latency, hit‑rate, cache depth, loop convergence).

### 4) Policy‑aware Hooks
- Hooks can bind prompts, tools, or *interpretive lenses* (e.g., a theology lens or a CUDA‑analysis lens). Policies are part of the lattice, not an afterthought.

---

## Architecture
```
app / agent
   │
   ├── PMLL Orchestrator
   │     ├── Lattice Store (graph + seals)
   │     ├── Loop Runner (recursive state, checkpoints)
   │     ├── Hook Registry (tools/prompts/policies)
   │     └── gRPC Adaptive Batcher (retrieval/tool I/O)
   │
   ├── Memory Backends
   │     ├── Vector / KV / Doc stores
   │     └── Versioned Artifacts (notes, code, PDFs)
   │
   └── Observability
         ├── Metrics (Prometheus)
         └── Dashboards (Grafana)
```
- **Orchestrator** traverses the lattice and manages loop state.
- **Hook Registry** binds domain hooks (e.g., *Sora‑DALL·E3* image/video pipeline hooks if present).
- **Batcher** enforces QoS, retries, and circuit‑breaking.


---

## Install & Setup

### 1) Clone the archive
```bash
git clone https://github.com/drQedwards/PMLL_archive.git
cd PMLL_archive
```

### 2) Unpack the AI Memory Loops bundle
If present:
```bash
unzip AI_memory_Loops.zip -d ./AI_memory_Loops
```
> The archive may include manifests, graph exports, glyph seals, and integration scripts.

### 3) (Optional) Python environment
If you’re extending with Python:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if provided
```

### 4) (Optional) Observability
Expose metrics locally:
```bash
# Prometheus scrape config (example)
# scrape_configs:
#   - job_name: 'pmll'
#     static_configs:
#       - targets: ['localhost:9095']
```
Point Grafana to your Prometheus and import the included dashboard JSON (if provided).

> **Note:** This repository is an archive. Implementation hooks may live in downstream repos where you wire the orchestrator to your app/agent runtime.


---

## Quickstart

### A. Run a lattice compile (conceptual example)
```bash
python tools/pmll_compile.py   --input data/ingest/   --seal out/seal.json   --graph out/lattice.graph.json
```
- Produces a **seal** (checksum + lineage)
- Produces a **graph export** suitable for traversal

### B. Traverse with a domain hook
```bash
python tools/pmll_traverse.py   --graph out/lattice.graph.json   --hook theology_exegesis   --query "Leviticus 18:22 – anti‑exploitation lens"
```

### C. Use the gRPC batcher (pseudo‑CLI)
```bash
pmll-batcher   --graph out/lattice.graph.json   --max-inflight 128   --qos high   --metrics :9095
```

> The above are **reference commands**. Replace with the actual scripts present in your bundle.


---

## Runtime hooks & gRPC batcher

- **Priority lanes:** `high`, `normal`, `bulk`
- **Backpressure:** token bucket w/ burst window
- **Retries:** exponential backoff for transient errors
- **Tracing:** loop‑ID and seal‑ID propagate through logs and spans
- **Safety:** policy hooks are *pre‑bound*; batcher rejects calls that violate policy before I/O

A typical request lifecycle:
1. Resolve **loop‑ID** and **seal‑ID**
2. Expand **context petals** (attention flower) → candidate nodes
3. Batch retrieval/tool calls with QoS
4. Merge responses into a **checkpoint**
5. Optionally **commit** the checkpoint back to the lattice


---

## Observability

**Key metrics** (Prometheus names are examples):
- `pmll_batch_inflight`
- `pmll_batch_latency_ms`
- `pmll_cache_hit_ratio`
- `pmll_lattice_traversal_depth`
- `pmll_loop_convergence_score`
- `pmll_policy_block_total`

**Dashboards**
- Throughput & Latency
- Lattice Health (node/edge counts, seal versions)
- Loop Convergence (stability of outputs across iterations)


---

## AI Memory Loops bundle

If `AI_memory_Loops/` is present after unzip, expect some/all of:
```
AI_memory_Loops/
├── manifests/
│   ├── lattice.manifest.json
│   └── seals/
├── exports/
│   ├── lattice.graph.json
│   └── embeddings/
├── hooks/
│   ├── theology_exegesis/
│   ├── cuda_analysis/
│   └── image_video_pipeline/  # Sora‑DALL·E adapters (optional)
└── tools/
    ├── pmll_compile.py
    ├── pmll_traverse.py
    └── pmll_batcher.py
```
> **Note:** This is a *canonical layout*. Your actual bundle may differ—use the manifest as source‑of‑truth.


---

## Data & Model Governance

- **Provenance:** Every artifact is stamped with a seal derived from content hash, timestamp, and lattice version.
- **Reproducibility:** Re-running the same inputs against the same lattice version must yield the same traversal and checkpoints.
- **Privacy/Security:** Treat seals and manifests as sensitive; avoid exposing private nodes. Hooks can enforce redaction at the edge.


---

## Roadmap
- [ ] Publish typed edge schema & policy DSL
- [ ] Release standardized gRPC proto for the batcher
- [ ] Add lattice diff/merge visualizer
- [ ] Export Grafana dashboards (JSON) with example Prometheus config
- [ ] Provide minimal reference implementation in Python + Rust
- [ ] Add CI checks that fail on unsealed graph changes


---

## Contributing
PRs welcome for docs, examples, and test fixtures. If contributing code, please link to the downstream runtime repository and include:
- Motivation & design notes
- Test plan and sample data
- Impact on seals/manifests (if any)

> For large changes, open an issue with a design sketch first.


---

## Citations & Further Reading
- Edwards, J. K. “Persistent Memory Logic Loop (PMLL)” — ResearchGate excerpts and community notes.
- AI Memory Loops release notes and sealed manifests (this archive).
- Background: recursive memory graphs, persistent knowledge stores, retrieval‑augmented generation (RAG), policy‑aware tool orchestration.

> See the **Releases** tab for the “PMLL_blockchain archive” and the `AI_memory_Loops.zip` asset if available.


---

## License
MIT. See `LICENSE` if present; otherwise treat this archive as MIT unless superseded by a downstream repo license.
