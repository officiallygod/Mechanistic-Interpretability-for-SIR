# MIS: Mechanistic Interpretability for SIR
### *Uncovering Implicit Object Sets in Robot Policies via Sparse Autoencoders*

[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-0284c7?style=for-the-badge&logo=github)](https://officiallygod.github.io/Mechanistic-Interpretability-for-SIR/)
[![Google Slides](https://img.shields.io/badge/Slides-Google%20Slides-FBBC04?style=for-the-badge&logo=googleslides&logoColor=white)](https://docs.google.com/presentation/d/1mFDo7Pkut4FRN9KIvGs0BECcrez3ElA9oxkdym2P1z8/edit?usp=sharing)
[![Miro Board](https://img.shields.io/badge/Miro-Architecture%20Board-050038?style=for-the-badge&logo=miro&logoColor=yellow)](https://miro.com/welcomeonboard/Q3BMR0hTZ09UditVS2hqUU1zZGRBZ0RnTlVVRGc4OGtyYzZieUkycDkxakRtYm5YWnZSZ0lodTQ2N0o4Nys4bG5ra0xHMEMrSVpaVTFCVkNPYllQd21yRE5KcTdYZDZxNXNFdFF3Mnd1ZWVHV1VHSFdxN25DNEtCOW1uVXdjNnp3VHhHVHd5UWtSM1BidUtUYmxycDRnPT0hdjE=?share_link_id=911869718076)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Lab](https://img.shields.io/badge/Intuitive%20Robots%20Lab-KIT-009682?style=for-the-badge)](https://intuitive-robots.net/)

> **Seminar & Praktikum:** Explainable Object Sets for Robot Policies (WS 2026/2027)  
> **Topic 2:** Mechanistic Interpretability with Object Sets (Sparse Autoencoders)  
> **Team (Group 4):** Allen & Luca  
> **Supervisor:** Paul Mattes  
> **Quick Links:** [🌐 Interactive Docs](https://officiallygod.github.io/Mechanistic-Interpretability-for-SIR/) • [🔍 Step-by-Step Tensor Walkthrough](https://officiallygod.github.io/Mechanistic-Interpretability-for-SIR/pipeline_explained.html) • [📊 Bi-Weekly Slides](https://docs.google.com/presentation/d/1mFDo7Pkut4FRN9KIvGs0BECcrez3ElA9oxkdym2P1z8/edit?usp=sharing) • [🧠 Miro Board](https://miro.com/welcomeonboard/Q3BMR0hTZ09UditVS2hqUU1zZGRBZ0RnTlVVRGc4OGtyYzZieUkycDkxakRtYm5YWnZSZ0lodTQ2N0o4Nys4bG5ra0xHMEMrSVpaVTFCVkNPYllQd21yRE5KcTdYZDZxNXNFdFF3Mnd1ZWVHV1VHSFdxN25DNEtCOW1uVXdjNnp3VHhHVHd5UWtSM1BidUtUYmxycDRnPT0hdjE=?share_link_id=911869718076) • [📅 Progress Logs](progress/)

---

## 1. Project Motivation & Research Question

In robot imitation learning, two paradigms currently exist:
1. **Explicit Object-Centric Policies (e.g. [SIR](https://github.com/intuitive-robots/sir)):** The visual scene is parsed into an explicit scene graph of objects. A selector prunes the graph down to an explicit "kept set" $K$ and claims intrinsic explainability. However, the companion ICLR 2027 analysis shows that explicit in-loop subset selections suffer from subtle leakage paths (e.g., softmax denominators normalizing over all nodes, survivor score weighting, presence counts) and often fail to simultaneously satisfy *Discovery*, *Removal*, and *Consequence*.
2. **End-to-End Image Policies (Pixel Baselines):** An image encoder (one ResNet-18 per camera) processes the entire visual frame directly into a diffusion action generator (MDT). These policies run without any object selection and remain opaque black boxes. In upstream SIR's 24-task RoboCasa benchmark, the image-only baseline reaches 12.93 % mean success vs. 16.60 % for SIR (`SIR_Model/robocasa_results.md`).

> **Official Topic Description:**  
> *"Train a sparse autoencoder on the activations of an image policy that has no object selection. Start with the ResNet-18 7×7 feature map before pooling. Label each feature by where it fires (activation map vs. object boxes from the simulator). The objects whose features are active at a step form a 'latent kept set'. Need to see if we take a whole object or only parts of objects (pixel level or object level). Direction: does a policy without explicit selection form an object set internally? Test the latent kept set with Discovery, Removal and Consequence, by switching features off during rollouts, and compare it with SIR's explicit kept set."*
>
> **Note on shapes:** "7×7" assumes 224 × 224 inputs. SIR's RoboCasa cameras are **128 × 128** (`SIR_Model/envs/robocasa/utils.py`), so in this codebase the pre-pooling `layer4` map is **4 × 4 × 512** (16 cells of 32 × 32 px).

### Key Research Dimensions:
1. **Implicit Object Set Discovery:** Does an end-to-end vision policy form an internal representation of objects without explicit supervisory selection?
2. **Object Granularity (Whole Objects vs. Parts vs. Pixels):** Do active SAE features correlate with entire object volumes (e.g., `drawer`, `mug`), functional interaction parts (e.g., `drawer handle`, `mug rim`, `gripper contact pads`), or localized pixel-level visual primitives?
3. **Causal Faithfulness (Discovery, Removal, Consequence):** When we switch features off during closed-loop execution, do excluded objects strictly cease to affect the robot's actions, and does performance drop predictably?
4. **Comparison with SIR:** Contrast this latent kept set with SIR's explicit, learned kept set to see if both approaches isolate the same objects or reveal similar dataset biases.

---

## 2. Methodology & Architecture

> 🗺️ **New to the SIR code?** Start with the [SIR Code Mind Map](docs/sir_code_mindmap.md). It covers every component of `SIR_Model/` and marks where our SAE pipeline plugs in.

```mermaid
flowchart TD
    subgraph S1["1. Visual Observation & Backbone (ResNet-18, one per camera)"]
        Cam["Agentview left / right camera\nRGB: 128 x 128 x 3"] --> Conv1["Conv1 + MaxPool\n32 x 32 x 64"]
        Conv1 --> L1["Layer 1 (BasicBlock)\n32 x 32 x 64"]
        L1 --> L2["Layer 2 (BasicBlock)\n16 x 16 x 128"]
        L2 --> L3["Layer 3 (BasicBlock)\n8 x 8 x 256"]
        L3 --> L4["Layer 4 (BasicBlock)\n4 x 4 x 512"]
        L4 --> Pool["Global Average Pooling + fc\nVector: 512-d"]
    end

    subgraph S2["2. Mechanistic Interpretability: Sparse Autoencoder (SAE)"]
        L4 -.->|"Tap 16 Spatial Cells (x_i in R^512)"| SAE_Enc["SAE Encoder: ReLU(W_enc · x + b)\nTop-K / L1 Sparsity"]
        SAE_Enc --> SparseF["Overcomplete Sparse Code\nf in R^4096 (L0 ≪ 4096)"]
        SparseF --> SAE_Dec["SAE Decoder: W_dec · f + b\nx_hat in R^512"]
        SparseF --> Grounding["Spatial Feature Heatmaps\nOverlap with Simulator Segmentation Masks"]
        Grounding --> LatentSet["Emergent Latent Kept Set K_latent\n(Mug, Drawer Handle, Gripper)"]
    end

    subgraph S3["3. Action Generator: MDT Diffusion Transformer"]
        Pool --> MDT_Enc["4-layer Transformer Encoder\n(Dim: 512, 16 Attention Heads)\nTokens: goal + left + right"]
        Goal["Language Instruction (CLIP)\n512-d Embedding"] --> MDT_Enc
        Noise["Noisy Action Chunk a_t^(k)\n[10 x 12] + Step k"] --> MDT_Dec["6-layer Transformer Decoder\n(Cross-Attention over Observations)"]
        MDT_Enc --> MDT_Dec
        MDT_Dec --> DDIM["10-Step DDIM Denoising"]
        DDIM --> Action["10-Step Action Chunk\n[10 x 12] end-effector actions"]
    end

    SAE_Dec -.->|"Causal Ablation / Activation Steering"| Pool
```

### The 6-Step Protocol:
1. **Run Rollouts with Trained Image Policy:** Execute an untouched ResNet-18 + MDT policy (`manager.image_modalities=[left,right]`, `manager.graph_modalities=[]`) and record the $4 \times 4 \times 512$ feature map at `layer4` before global average pooling. The cameras are 128 × 128 px, so each of the 16 cells covers a 32 × 32 px patch.
2. **Train a Sparse Autoencoder (SAE):** Train an overcomplete dictionary ($M = 4096$, $8\times$ expansion) with Top-K or $L_1$ sparsity on these activation vectors to disentangle polysemantic superposition into monosemantic features.
3. **Label Features via Spatial Grounding:** Map active features across the $4 \times 4$ grid into 2D heatmaps and compare them with the simulator's per-pixel segmentation masks (`robot0_agentview_*_segmentation_class`), which SIR already uses to build its scene graphs.
4. **Form the Latent Kept Set ($K_{latent}$):** Identify the set of objects whose grounded features are firing at each policy call (once per 10-step action chunk).
5. **Causal Intervention & Condition Testing:** Clamping or zeroing out features during rollout to test *Discovery*, *Removal*, and *Consequence*.
6. **Benchmark Against SIR:** Contrast $K_{latent}$ with SIR's explicit kept set to verify whether the policy learns human-expected objects or exploits dataset biases (e.g., positional biases, background shortcuts).

---

## 3. The Three Evaluation Criteria

| Condition | Core Question | Experimental Verification | Status |
| :--- | :--- | :--- | :--- |
| **Discovery (D-C)** | Did the model discover these objects on its own, or was it imposed by annotations? | Permutation test: The pixel policy is trained without any object supervision or bounding-box loss, passing by construction. We measure whether emergent features align with task-critical objects above chance. | **Measured** |
| **Removal (R-C)** | Do excluded objects truly have zero influence on the downstream action? | Causal feature ablation: Zero out SAE features corresponding to excluded objects ($f_j = 0$), reconstruct $\hat{x}$, and verify action invariance during closed-loop rollout. | **Measured via Ablation** |
| **Consequence (C-C)** | Does the selection matter beyond a random subset of the same size? | Closed-loop policy evaluation: Test whether ablating $K_{latent}$ causes task failure compared to matched random feature ablations. | **Measured on Success** |

---

## 4. Repository Structure

```
Mechanistic-Interpretability-for-SIR/
├── .github/workflows/
│   └── deploy-pages.yml       # Automated GitHub Pages CI/CD deployment
├── docs/                      # Interactive GitHub Pages documentation
│   ├── index.html             # Academic project portal with interactive diagrams
│   ├── pipeline_explained.html # Step-by-step tensor walkthrough & vector mappings
│   ├── .nojekyll              # Disables Jekyll processing on GitHub Pages
│   ├── architecture_deep_dive.md # Detailed tensor dimensions & layer specifications
│   └── sir_code_mindmap.md    # Mind map of the SIR codebase + MIS integration placeholders
├── papers/                    # Foundational research literature
│   ├── CVPR_graph_paper_2026.pdf # SIR: Structured Image Representations (CVPR 2026)
│   ├── Closing_the_loop_SIR_ICLR_2027-1.pdf # Analysis of subset explanations (ICLR 2027)
│   └── README.md              # Context & literature summaries
├── progress/                  # Bi-weekly progress updates & logs
│   ├── README.md              # Progress tracker & meeting cadence
│   └── cycle_01_kickoff.md    # Kickoff & architecture blueprint notes
├── notes/                     # Team meeting logs and discussion points
│   └── questions_to_paul.md   # Open research questions for supervisor
├── src/                       # Core Mechanistic Interpretability implementation
│   ├── __init__.py            # Package root
│   ├── activation_recorder.py # PyTorch forward hooks for ResNet-18 layer4 activations
│   ├── sae.py                 # Top-K / L1 Sparse Autoencoder model
│   ├── grounding.py           # Spatial heatmap projection & bounding-box IoU
│   └── causal_intervention.py # Feature clamping & closed-loop ablation hooks
├── SIR_Model/                 # Vendored upstream SIR (intuitive-robots/sir @ 49321d0)
│   ├── overview.md            # Upstream design notes (sparsification, VGIB, eval knobs)
│   └── robocasa_results.md    # Upstream results: SIR vs. VGIB vs. Image-only
├── .env                       # Local environment variables & API keys (git-ignored)
├── .gitignore                 # Exclusion rules for weights (*.pth), caches, and data
└── README.md                  # Project overview & roadmap (this file)
```

---

## 5. Bi-Weekly Progress & Milestone Tracking

Following the seminar organization (meeting every two weeks with Paul Mattes), each cycle answers:
1. *What was accomplished in the last two weeks?*
2. *What are the next concrete steps?*
3. *Current blockers & hypotheses.*

* Check out [`progress/README.md`](progress/README.md) for the complete timeline.
* Active slide deck: [Google Slides](https://docs.google.com/presentation/d/1mFDo7Pkut4FRN9KIvGs0BECcrez3ElA9oxkdym2P1z8/edit?usp=sharing)
* Live concept board: [Miro](https://miro.com/welcomeonboard/Q3BMR0hTZ09UditVS2hqUU1zZGRBZ0RnTlVVRGc4OGtyYzZieUkycDkxakRtYm5YWnZSZ0lodTQ2N0o4Nys4bG5ra0xHMEMrSVpaVTFCVkNPYllQd21yRE5KcTdYZDZxNXNFdFF3Mnd1ZWVHV1VHSFdxN25DNEtCOW1uVXdjNnp3VHhHVHd5UWtSM1BidUtUYmxycDRnPT0hdjE=?share_link_id=911869718076)

---

## 6. Development & GitHub Pages Deployment

### Enabling GitHub Pages:
The repository includes an automated GitHub Action (`.github/workflows/deploy-pages.yml`) that publishes `docs/` to GitHub Pages on every push to `main`.

To ensure GitHub Pages is enabled in your repository:
1. Go to **Settings** $\to$ **Pages** in the GitHub repository.
2. Under **Build and deployment** $\to$ **Source**, select **GitHub Actions**.
3. Pushing any update to `main` will automatically build and deploy the interactive documentation portal!

### Local Collaboration:
```bash
# Pull latest changes from teammates
git pull origin main

# Stage and commit new changes
git add .
git commit -m "feat: description of work"
git push origin main
```
