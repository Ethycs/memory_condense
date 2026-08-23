# Unified Framework: QK/OV Circuits, Graph Transformers, and Latent Concept Routing

This document summarizes the comprehensive exploration of mapping **Query-Key (QK)** and **Output-Value (OV)** attention circuits from sequence-based Large Language Models (LLMs) onto graph topologies, escalating to highly scalable **Latent Concept Activation Vector (CAV)** routing systems.

---

## 1. Core Mechanics of QK and OV Circuits
In transformer architectures, attention heads can be functionally split into two complementary systems that track the movement and transformation of concept vectors:

*   **The QK Circuit (Where to Look):** Computes the inner dot-product alignments between token/node features ($W_Q$ and $W_K$). It operates as a soft, content-dependent router determining the attention weight matrix.
*   **The OV Circuit (What to Move):** Dictates the structural payload. It reads the raw attributes from the source position, filters or transforms them via $W_V$ and $W_O$, and writes the final concept vector directly into the destination's residual stream.

---

## 2. Graph Transformers: The "LLM Conversion" for Graphs
A **Graph Transformer (GT)** explicitly adapts this sequence-based LLM blueprint to non-linear graph structures:

*   **Sequence Tokens $\rightarrow$ Nodes:** Instead of words, tokens represent node embeddings (e.g., text chunks, molecules, or profiles).
*   **Positional Encodings $\rightarrow$ Structural Encodings:** Temporal order is replaced with graph geometry metrics, such as **Laplacian Eigenvectors** (global graph shape) and **Random Walk Positional Encodings (RWPE)** (local neighborhood density).
*   **The Attention Paradigm:** Rather than strict localized propagation, GTs use global or scaled-local attention, turning physical graph wiring into a soft inductive bias rather than a rigid boundary.

### Comparative Architectural Analysis
The table below contrasts Graph Transformers with traditional Graph Attention Networks (GATs):

| Feature | Graph Attention Network (GAT) | Graph Transformer (GT) |
| :--- | :--- | :--- |
| **Scoring Mechanism** | **Additive:** Uses a shared linear layer after concatenating or calculating differences. | **Dot-Product:** Natively tracks multidimensional alignment via distinct $W_Q$ and $W_K$ matrices. |
| **Receptive Field** | **Strictly Local:** Limited to 1-hop physical structural neighbors. | **Global / Hybrid:** Allows any node to see any node globally, biased by distance encodings. |
| **Information Payload** | Compresses and shifts isotropic, static entire node vectors. | Extracts, projects, and maps filtered concept vectors via the **OV Circuit**. |
| **Long-Range Execution** | Poor; prone to **over-smoothing** and **over-squashing** past 4–5 hops. | Perfect; skips topological bottlenecks via direct global routing in a single layer. |

---

## 3. The Scalability Wall & Chunks-on-Nodes
When graphs are scaled to enterprise-grade levels (e.g., Text-Graph Transformers or Graph-RAG systems where nodes hold extensive text chunks), computing unconstrained global attention runs into a severe memory boundary.

A fully unconstrained QK circuit requires an $\mathcal{O}(N^2)$ matrix expansion. For a graph with $1,000,000$ nodes, the resulting attention matrix contains **1 trillion elements**, resulting in immediate hardware Out-of-Memory (OOM) errors.

---

## 4. Bypassing the Wall: Pure Attention CAV Routing
To retain expressive global routing without hitting the quadratic memory wall, the architecture can pivot to **Latent Concept Routing**. This mimics episodic memory indexing during human recall.

Instead of forcing a direct node-to-node ($N \times N$) matrix expansion, the system processes two sequential, highly efficient cross-attention operations using a small, fixed bottleneck of $K$ learned latent concept tokens ($K \ll N$, e.g., $K=64$).

```
[Phase 1: Concept Extraction]
 Node Features [N x D] ───(Cross Attention)───► Learned Latent Tokens [K x D]
 Matrix Complexity: O(K * N)

[Phase 2: Concept Reinjection]
 Learned Latent Tokens [K x D] ───(Cross Attention)───► Node Features [N x D]
 Matrix Complexity: O(N * K)
```

### Mechanistic Operational Phases

1.  **The Extraction Head (The Compressed QK/OV Mapping):**
    The $K$ latent tokens act as global **Queries**, while the $N$ nodes act as **Keys** and **Values**. The QK circuit measures how strongly each node aligns with a global latent concept. The OV circuit then pools the semantic attributes of the matching nodes into the latent spaces.
2.  **The Reinjection Head (The Directional Steering Phase):**
    The $N$ nodes act as **Queries**, while the filled $K$ concept tokens act as **Keys** and **Values**. The network maps the macro-concepts back onto individual nodes, injecting a directional steering force directly into each node's local residual stream.

### Architectural Performance Footprint
This configuration scales the computational complexity downward from quadratic to entirely linear, enabling true global context processing within strict hardware limitations:

*   **Vanilla Graph Transformer Complexity:** $\mathcal{O}(N^2 \cdot D)$
*   **Pure Attention CAV Head Complexity:** $\mathcal{O}(N \cdot K \cdot D)$

Through this bottleneck framework, the attention heads naturally specialize over training—mapping local documents into compressed episodic representations in the extraction phase, and executing associative topological recall during the reinjection phase.
