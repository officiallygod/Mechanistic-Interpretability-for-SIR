# Reference Literature & Foundation Papers

This directory archives the primary reference papers from the **Intuitive Robots Lab (KIT)** foundational to our research topic:

---

## 1. SIR: Structured Image Representations for Explainable Robot Learning
* **Authors:** Paul Mattes, Jan Schwab, Jens Bosch, Maximilian Xiling Li, Nils Blank, Minh-Trung Tang, Moritz Haberland, Rudolf Lioutikov
* **Venue:** CVPR 2026
* **File:** [`CVPR_graph_paper_2026.pdf`](CVPR_graph_paper_2026.pdf)
* **Core Contribution:**
  - Proposes Structured Image Representations (SIR) using Scene Graphs (SGs) as an intermediate representation in Goal-Conditioned Imitation Learning (GCIL).
  - Learns to prune the scene graph end-to-end via a transformer-decoder module (`FiLMDecoder`), retaining a task-relevant sub-graph passed to a Graph Attention Network (`GATv2`) and action generator (`MDT`).
  - Demonstrates that the sparsified graph can reveal dataset biases (spurious correlations, positional biases).

---

## 2. Analyzing When a Learned Subset of Objects Explains a Robot Policy: Discovery, Removal, Consequence
* **Authors:** Anonymous / Under Review
* **Venue:** ICLR 2027
* **File:** [`Closing_the_loop_SIR_ICLR_2027-1.pdf`](Closing_the_loop_SIR_ICLR_2027-1.pdf)
* **Core Contribution:**
  - Formalizes **In-Loop Subset Explanations (ILSE)** and defines the three necessary conditions for subset explanations:
    1. **Discovery (D-C):** The selection must be discovered by the model rather than imposed by ground-truth annotation lists.
    2. **Removal (R-C):** Excluded objects must be strictly absent from the downstream computation ($f_\theta^{(p)}(O_t, m_t, c) = f_\theta^{(p)}(C(S_t), 1, c)$).
    3. **Consequence (C-C):** The selection must matter beyond a matched random subset of the same size.
  - Shows that previous explicit architectures leak information through standard graph operators, and closing these leaks causes a significant drop in policy success.
  - Motivates our investigation (**Topic 2**): Rather than imposing an explicit graph bottleneck, does an untouched end-to-end vision policy form an *implicit* kept set of objects internally?
