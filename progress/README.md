# Research Progress & Session Log

This directory tracks dated working sessions, bi-weekly supervisor milestones, and hardware constraints for **MIS** (Mechanistic Interpretability for SIR) at the Intuitive Robots Lab, KIT.

---

## Active Hardware Notice
- **HoreKa2 HPC Cluster:** Unavailable until **December 04, 2026**.
- **Pre-Dec 04 Window (Nov 27 to Dec 03):** Literature close reading, P1-P11 comparison matrix, hook/SAE code verification on synthetic tensors, and SLURM script preparation (zero GPU dependency).
- **Post-Dec 04 Window (From Dec 04):** Single-task `CloseDrawer` policy rollouts, per-camera `layer3`/`layer4` activation dumping, and SAE training runs on HoreKa2.

---

## Meeting Resources & Links
- **Presentation Slides (Bi-Weekly Meetings):** [Google Slides](https://docs.google.com/presentation/d/1mFDo7Pkut4FRN9KIvGs0BECcrez3ElA9oxkdym2P1z8/edit?usp=sharing)
- **Brainstorming & Architecture Board:** [Miro Board](https://miro.com/welcomeonboard/Q3BMR0hTZ09UditVS2hqUU1zZGRBZ0RnTlVVRGc4OGtyYzZieUkycDkxakRtYm5YWnZSZ0lodTQ2N0o4Nys4bG5ra0xHMEMrSVpaVTFCVkNPYllQd21yRE5KcTdYZDZxNXNFdFF3Mnd1ZWVHV1VHSFdxN25DNEtCOW1uVXdjNnp3VHhHVHd5UWtSM1BidUtUYmxycDRnPT0hdjE=?share_link_id=911869718076)
- **Project Portal & Documentation:** [GitHub Pages](https://officiallygod.github.io/Mechanistic-Interpretability-for-SIR/)

---

## Dated Session Log Index (Newest First)

Every working session or milestone uses an ISO date-stamped file (`YYYY-MM-DD_topic.md`) recording what was completed, key decisions, next steps, and blockers.

| Date | Cycle | Category | Owner | Summary | Log File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2026-11-27** | Cycle 02 | Literature & Planning | Allen & Luca | Expanded literature base, filtered to top 11 core papers (P1-P11), read & understood pixel/part vs. whole-object vision SAEs; re-sequenced Cycle 03 around HoreKa2 downtime until Dec 04. | [2026-11-27_literature_curation.md](2026-11-27_literature_curation.md) |
| **2026-11-26** | Cycle 02 | Supervisor Sync | Allen & Luca | Bi-weekly sync with Paul Mattes (13:00 to 14:00): presented intro deck, spatial ResNet-18 tap strategy, and open questions. | *Synced in Slides* |
| **2026-11-19** | Cycle 02 | Codebase & Architecture | Allen & Luca | Synced `SIR_Model/` to upstream (`49321d0`), corrected `layer4` shape to `4x4x512` for `128x128` cameras, and added `docs/sir_code_mindmap.md`. | [cycle_01_kickoff.md](cycle_01_kickoff.md) (Addendum) |
| **2026-11-13** | Cycle 01 | Kickoff & Scope | Allen & Luca | Kickoff meeting with Paul Mattes: finalized Topic 2, analyzed SIR & ILSE criteria (D-C, R-C, C-C), and scaffolded `src/` modules. | [cycle_01_kickoff.md](cycle_01_kickoff.md) |
| **2026-11-05** | Cycle 01 | Infra / Repo | Allen & Luca | Initialized team repository, imported upstream `SIR_Model/`, and configured GitHub Pages CI/CD deployment. | [README.md](README.md) |

---

## Bi-Weekly Supervisor Schedule

**Working Schedule & Cadence:**
- **Weekly Team Working Session:** Every Friday from 08:00 AM
- **Bi-Weekly Supervisor Meeting:** Thursdays 13:00 to 14:00 (Next sync: Dec 10, 2026 at 13:00 to 14:00)
- **HoreKa2 Maintenance Window:** Offline until Dec 04, 2026
- **Academic Recess (Winter Break):** Dec 24, 2026 to Jan 06, 2027

| Cycle | Date Range | Focus Area | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Meeting 1** | Kickoff: Nov 13, 2026 | Topic Definition, Initial Setup & Scope Alignment | **Completed** | [Cycle 01 Log](cycle_01_kickoff.md) |
| **Meeting 2** | Nov 14 to Nov 27, 2026 | Codebase Sync, Literature Expansion & Top-11 Paper Reading (Sync: Nov 26) | **Completed** | [Nov 27 Log](2026-11-27_literature_curation.md) |
| **Meeting 3** | Nov 28 to Dec 11, 2026 | **Phase A (Nov 28 to Dec 03, Offline):** P1-P11 Matrix & Local Hook Dry-Run. **Phase B (Dec 04 to Dec 11, HoreKa2 GPU):** RoboCasa Rollouts & Activation Capture (Sync: Dec 10) | **Next** | HoreKa2 resumes Dec 04 |
| **Meeting 4** | Dec 12 to Dec 23, 2026 | HoreKa2 SAE Training (TopK/BatchTopK), Spliced Fidelity Check & Seminar Outline Sign-Off | Scheduled | Pre-recess milestone |
| **Break** | Dec 24, 2026 to Jan 06, 2027 | University Winter Holiday Recess (Optional Section 2 Drafting) | Scheduled | *Recess* |
| **Meeting 5** | Jan 07 to Jan 21, 2027 | Latent Kept-Set Grounding, Closed-Loop D-R-C Ablations & SIR Comparison (Sync: Jan 14) | Scheduled | Core results |
| **Meeting 6** | Jan 22 to Feb 2027 | Synthesis, Final Seminar (6+p) & Praktikum (8+p) Reports, 20-min Presentation | Scheduled | Final delivery |

---

## Reusable Session Log Template

To log a new session, create `progress/YYYY-MM-DD_short_topic.md` with:
1. **Header Metadata:** Date (`YYYY-MM-DD`), Cycle, Category (`Literature`, `Code`, `Experiments`, `Sync`, `Infra`), Team/Owner, Cluster Status.
2. **Summary of Work Done:** Concrete outputs and verified findings.
3. **Key Decisions & Takeaways:** Architectural or experimental choices locked in.
4. **Next Concrete Steps:** Actionable items tied to dates and compute availability.
5. **Blockers & Open Questions:** Issues to resolve or bring to the next supervisor sync.
