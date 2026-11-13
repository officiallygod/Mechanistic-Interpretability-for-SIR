# Meeting 1 Progress Log: Project Kickoff & Architecture Blueprint

*Date: November 13, 2026*  
*Team: Allen & Luca (Group 4)*  
*Presentation Slides:* [Google Slides](https://docs.google.com/presentation/d/1mFDo7Pkut4FRN9KIvGs0BECcrez3ElA9oxkdym2P1z8/edit?usp=sharing)  
*Miro Board:* [Miro](https://miro.com/welcomeonboard/Q3BMR0hTZ09UditVS2hqUU1zZGRBZ0RnTlVVRGc4OGtyYzZieUkycDkxakRtYm5YWnZSZ0lodTQ2N0o4Nys4bG5ra0xHMEMrSVpaVTFCVkNPYllQd21yRE5KcTdYZDZxNXNFdFF3Mnd1ZWVHV1VHSFdxN25DNEtCOW1uVXdjNnp3VHhHVHd5UWtSM1BidUtUYmxycDRnPT0hdjE=?share_link_id=911869718076)

---

## 1. What was done in the last two weeks:
- **Topic Selection & Alignment:** Finalized selection of **Topic 2: Mechanistic Interpretability with Object Sets**.
- **Theoretical Formalization:** Analyzed the companion ICLR 2027 paper on In-Loop Subset Explanations (ILSE), formalizing the **Discovery (D-C)**, **Removal (R-C)**, and **Consequence (C-C)** evaluation criteria.
- **Architecture Reverse-Engineering:** Mapped out the existing policy architecture in `SIR_Model`:
  - Image Backbone: Pretrained ResNet-18 producing a $7 \times 7 \times 512$ feature map at `layer4`.
  - Action Generator: Multimodal Diffusion Transformer (MDT) with 10-step action chunking and CLIP goal conditioning.
- **Mechanistic Interpretability Blueprint:**
  - Designed the Sparse Autoencoder (SAE) tap into the 49 spatial cells of `layer4`.
  - Formulated the research direction for **Whole Objects vs. Object Parts** (pixel-level vs affordance-level vs whole-object grounding).
- **Workspace & Collaboration Setup:**
  - Clean repository isolation with GitHub CI/CD for GitHub Pages.
  - Initialized modular Python implementations in `src/` (`activation_recorder.py`, `sae.py`, `grounding.py`, `causal_intervention.py`).

---

## 2. Next Steps (Upcoming 2 Weeks):
1. **Activation Recording Pipeline:** Run evaluation rollouts on RoboCasa (`CloseDrawer`, `CloseSingleDoor`) with the image policy and dump the $7 \times 7 \times 512$ activations.
2. **SAE Training & Tuning:** Train an initial overcomplete SAE ($M = 4096$) with Top-K sparsity ($K = 32$) on the collected activations and measure reconstruction MSE vs. $L_0$ sparsity.
3. **Heatmap Generation:** Visualize the spatial activation heatmaps of the top active SAE features across the camera frames.
4. **Questions for Paul:** Discuss questions in `notes/questions_to_paul.md` regarding static vs. dynamic object grounding and temporal attention dynamics.
