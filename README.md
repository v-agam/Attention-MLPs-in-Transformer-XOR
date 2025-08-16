# The Role of Attention and MLPs in Transformer XOR Computation

This repository contains code, experiments, and analysis for studying how small Transformer models learn and compute the **XOR function** on bit sequences. We extend mechanistic interpretability techniques from the well-studied **4-bit XOR** task to the more challenging **7-bit XOR**, providing a detailed account of the internal circuits, attention head roles, and neuron selectivity.  

---

## 🚀 Research Motivation  
XOR is a classic benchmark for testing nonlinear computation in neural networks. Despite its simplicity, XOR requires mechanisms beyond linear separability, making it an ideal test case for probing **how Transformers actually compute**.  

- Prior work has characterized **4-bit XOR**, showing a division of labor between **attention (copying inputs)** and **MLPs (introducing nonlinearity)**.  
- Here, we investigate whether these mechanisms scale to **7-bit XOR**, where the model must generalize over longer offsets.  

---

## 🧪 Experiments  

We trained a 1-layer, 2-head Transformer on the 7-bit XOR task and analyzed it using mechanistic interpretability tools:  

1. **Attention Analysis**  
   - Head 0 emerges as the primary **routing head**, attending from positions 7–13 back to 0–6.  
   - Path patching and ablations confirm that this head is causally necessary (Head 0 ablation → acc ≈ 0.379).  
   - Head 1 plays a secondary/auxiliary role (ablation → acc ≈ 0.838).  

2. **Linear Probes**  
   - Early layers encode only local bits (`bj`).  
   - Attention injects the far bit (`bi`), but XOR remains inseparable.  
   - At the MLP layer, XOR becomes linearly separable (probe acc = 1.0).  
   - Final residual streams mix both raw inputs and the XOR feature.  

3. **Neuron Selectivity**  
   - MLP neurons specialize into **00, 01, 10, 11 detectors**.  
   - XOR is reconstructed as a linear combination of pro-XOR (01,10) and anti-XOR (00,11) neuron groups.  
   - Compared to 4-bit XOR, the clustering is less sharp, suggesting more distributed representations in 7-bit.  

4. **Ablation Studies**  
   - Random-K vs. Top-K ablations show that a small subset of neurons (≈16 out of 256) suffices for XOR.  
   - Accuracy remains robust until ≈128 neurons are removed, then collapses.  
   - Overlap-based rankings fail, confirming XOR is not aligned to a single direction but distributed across subspaces.  

---

## 📊 Key Findings  

- **Stable Mechanism Across Tasks:** The **attention → MLP → XOR** pipeline discovered in 4-bit tasks persists in the 7-bit model.  
- **Head Assignment is Flexible:** In 4-bit, Head 1 was the router; in 7-bit, Head 0 takes that role. This indicates head specialization is determined by training dynamics, not architecture.  
- **Neuron Efficiency:** XOR computation is highly redundant yet compressible — only a handful of neurons are strictly necessary.  
- **Scalable Circuit Motif:** Despite longer offsets, the Transformer reuses the same compositional circuit structure.  

---

## 🔬 Comparison: 4-bit vs. 7-bit XOR  

| Feature                  | 4-bit XOR              | 7-bit XOR              |
|---------------------------|------------------------|------------------------|
| Routing Head              | Head 1                 | Head 0                 |
| XOR Separability          | At MLP                 | At MLP                 |
| Neuron Selectivity        | Clean 00/01/10/11      | More overlapping groups|
| Mechanism Structure       | Attention → MLP → XOR  | Attention → MLP → XOR  |

## PCA-based Representation Tracking

We fit a PCA basis on activations from the final training checkpoint to ensure stable projection axes. All earlier checkpoints are projected into this fixed space, with silhouette scores tracking label separation and locked-axis GIFs visualizing how representations cluster and evolve during training.
---

## 📂 Repo Structure  

```text
.
├── data/              # Training datasets (XOR bitstrings)
├── models/            # Saved Transformer checkpoints
├── notebooks/         # Jupyter notebooks for analysis
├── figures/           # Visualizations (attention maps, probes, ablations)
├── src/               # Core code for training + interpretability
└── README.md          # This file
```


---

## 🛠️ Methods Used  

- Transformer training with small architectures  
- Path patching for causal tracing  
- Linear probes for feature separability  
- Attention pattern visualization  
- Neuron selectivity and ablation analysis  

---

## 📌 Citation  

If you use this work or build upon it, please cite:  

```bibtex
@misc{xor-transformer-interpretability,
  author = {Agam Vuppulury, Gargi Rathi, Srujananjali Medicharla, Vaaruni Desai},
  title = {The Role of Attention and MLPs in Transformer XOR Computation},
  year = {2025},
  howpublished = {\url{https://github.com/Attention & MLPs in Transformer XOR}}
}

