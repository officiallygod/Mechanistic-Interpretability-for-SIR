# Session Log: Literature Expansion, Top-11 Selection & HoreKa2 Schedule Adjustment

- **Date:** 2026-11-27
- **Cycle:** Cycle 02 (Closeout)
- **Category:** Literature, Synthesis & Planning
- **Team:** Allen & Luca (Group 4)
- **Cluster Status:** HoreKa2 unavailable until 2026-12-04 (all GPU jobs queued post-Dec 04)

---

## 1. Summary of Work Done
- **Expanded Literature Base:** Conducted a targeted search for spatial vision and CNN Sparse Autoencoder papers beyond LLM/VLA literature, focusing specifically on whether visual encoders represent scenes **pixel/part-wise vs. whole-object-wise**.
- **Filtered to Top 11 Core Papers (`P1` to `P11`):** Pruned lower-relevance LLM-only removal papers while retaining Luca's verified selections (`P1`, `P7`, `P8`), shared foundations (`P2`, `P3`, `P4`), and adding high-relevance vision SAE papers (`P5`, `P6`, `P9`, `P10`, `P11`).
- **Deep Reading & Understanding:** Read and synthesized the selected top papers against our exact RoboCasa `128x128` ResNet-18 (`layer3` 8x8x256 and `layer4` 4x4x512) setup:
  - **Concept-SAE (`P7`) & ConvNeXt Spatial SAE (`P8`):** Confirmed spatial fusion risks in deeper ResNet layers and the necessity of set-level ablations across all object cells with random-region / random-feature controls.
  - **Concept Cones (`P9`), Matryoshka SAE (`P10`), & Universal SAE (`P11`):** Identified quantitative containment metrics, nested coarse-to-fine granularity (whole object vs. local part), and cross-view alignment frameworks for left + right RoboCasa cameras.
- **Schedule Alignment for HoreKa2 Downtime:** Re-sequenced Cycle 03 so all non-GPU tasks run from Nov 28 to Dec 03, and GPU-heavy rollouts, activation dumping, and SAE training begin immediately once HoreKa2 returns on Dec 04.

---

## 2. Key Decisions & Takeaways
1. **Granularity Test Plan:** Recording both `layer3` (8x8) and `layer4` (4x4) in a single rollout pass on HoreKa2 (post-Dec 04) directly tests part-level vs. whole-object-level emergence.
2. **Pre-Dec 04 Zero-Idle Plan:** Use the HoreKa2 maintenance window (Nov 28 to Dec 03) to complete the P1-P11 Seminar comparison matrix, finish appendix reading, and dry-run `activation_recorder.py` and `sae.py` locally on synthetic `[B, 512, 4, 4]` tensors.
3. **Structured Date-Stamped Logging:** Adopted ISO date-stamped session logs (`YYYY-MM-DD_topic.md`) indexed in `progress/README.md` and mirrored in the portal's filterable Research Log view.

---

## 3. Next Concrete Steps
- **Nov 28 to Dec 03 (Local / No GPU):**
  1. Build the structured P1-P11 comparison matrix (architecture, tap layer, SAE variant, grounding method, D-R-C evaluation).
  2. Verify forward/backward hook shapes and SLURM batch scripts for HoreKa2 on dummy tensors.
- **From Dec 04 (HoreKa2 Online):**
  1. Run single-task `CloseDrawer` image policy rollouts (left + right cameras) and record baseline success rate.
  2. Dump `layer3` and `layer4` spatial activation tensors per camera and validate storage/throughput ahead of the Dec 10 supervisor sync.

---

## 4. Blockers & Open Questions
- **Blocker (External, Resolved Dec 04):** HoreKa2 HPC cluster maintenance prevents GPU policy training and RoboCasa rollout dumping until Dec 04, 2026.
- **Open Question for Dec 10 Sync:** Confirm whether to run a small pilot of Matryoshka SAE (`P10`) or Universal SAE (`P11`) alongside BatchTopK (`P5`) on `CloseDrawer` once baseline activations are captured.
