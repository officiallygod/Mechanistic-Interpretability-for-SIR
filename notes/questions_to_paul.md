# Questions for Paul Mattes (Supervisor)

*Seminar & Praktikum · Intuitive Robots Lab, KIT*  
*Topic 2: Mechanistic Interpretability with Object Sets*  
*Group 4: Allen & Luca*

---

## 1. Static vs. Dynamic Objects & Spurious Correlations
**Question:**
> What do the experiments look like? Are there static objects? If not, then why does the model consistently include unrelated objects, such as the `Oven`, `FramedWindow`, and `Microwave`?

**Context & Background:**
In the SIR paper (Section 5.4, Fig. 5c for `CloseDrawer`), the TopK sparsified model achieves high task success (~81%) while consistently attending to unrelated fixtures like the oven, window, and microwave instead of the target drawer (which is only included 11% of the time).
* Is this because static fixtures provide stable global reference frames / camera calibration cues?
* Or does the dataset contain spurious correlations where fixed background objects leak information about the drawer state or trajectory?

---

## 2. Trajectory Horizon & Attention Dynamics Over Time
**Question:**
> How long is the predicted trajectory? Does the attention to nodes change throughout the rollout?

**Context & Background:**
* The MDT action generator predicts an action chunk of 10 steps ($H = 10$).
* In the current SIR setup, does the sparsification module recalculate the kept set at every single observation frame $t$, or does it stay invariant?
* In mechanistic interpretability, as the robot arm approaches the target object vs. when it is moving back, does the latent SAE feature activation shift dynamically from approach targets to gripper state?

- Why are we not combining the camera views beforehand and only combine the embeddings after the GNN passing? one view might be completely wrong, or even both, as they both might miss some important object. What cameras/views are we using? why(did you test the different setups)?