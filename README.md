# The Role of Attention and MLPs in Transformer XOR Computation

This repository contains code, experiments, and analysis for studying how small Transformer models learn and compute the **XOR function** on bit sequences. We apply mechanistic interpretability techniques to **4-bit XOR** task and also to the **7-bit XOR** task, providing a detailed account of the internal circuits, attention head roles, and neuron selectivity while also identifying the scalable learnings as the input bit size grows.  

---

## 🚀 Research Motivation  
XOR is a classic benchmark for testing nonlinear computation in neural networks. Despite its simplicity, XOR requires mechanisms beyond linear separability, making it an ideal test case for probing **how Transformers actually compute**.  

- In this work, we first analyze how a Transformer learns the **4-bit XOR** task, revealing a clear division of labor between **attention (copying inputs)** and **MLPs (introducing nonlinearity)**.  
- We then extend the same framework to the more challenging **7-bit XOR**, to test whether these mechanisms remain stable and how they adapt when the model must generalize over longer offsets.  

---

## 🧪 Experiments  

We conducted a series of controlled experiments on Transformer models to study how they learn the XOR function.  

1. **4-bit XOR (Baseline Study)**  
   - The 2-layer, 2-head Transformer learns XOR by splitting roles:  
     - **Attention heads** copy the relevant input bits across positions.  
     - **MLP neurons** introduce the nonlinearity required for XOR.  
   - Linear probes confirm that XOR only becomes separable in the MLP layer.  
   - Neuron selectivity emerges with clear **00, 01, 10, 11 detectors**, reconstructing XOR as a linear combination of pro-XOR and anti-XOR neuron groups.  

2. **7-bit XOR (Extended Study)**  
   - Scaling up to 7 bits, Head 0 emerges as the primary **routing head**, attending from later positions back to the earlier inputs.  
   - Path patching and ablations confirm causal necessity: removing Head 0 collapses accuracy (≈0.379), while Head 1 is only partially necessary (≈0.838).  
   - Linear probes show the same pattern as in 4-bit: raw bits are linearly encoded early, XOR becomes separable only at the MLP layer.  
   - Neuron selectivity is less cleanly separated than in 4-bit, with more overlap among the groups, suggesting a **more distributed representation**.  
   - Ablation studies show XOR remains robust even when many neurons are silenced; a small subset (~16 of 256 neurons) suffices for correct computation.  

---

## 📊 Key Findings  

- **Shared Mechanism Across Scales:** Both 4-bit and 7-bit models rely on the same compositional circuit: **attention copies → MLP nonlinearity → XOR output**.  
- **Flexible Head Specialization:** In 4-bit, Head 1 takes the routing role, while in 7-bit it shifts to Head 0 — showing that head assignment is flexible and training-dependent.  
- **Neuron Efficiency and Redundancy:** XOR requires very few specialized neurons, though the network recruits many more, indicating redundancy and robustness.  
- **Representation Sharpness vs. Distribution:**  
  - 4-bit XOR produces **cleanly clustered 00/01/10/11 neurons**.  
  - 7-bit XOR produces **more overlapping, distributed groups**, suggesting higher complexity in scaling.  
- **Generalization of Circuit Motif:** Despite longer offsets and higher task complexity, the **same structural motif** discovered in 4-bit scales naturally to 7-bit.

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
├── Code/            # Core code for training + interpretability
├── Figures/         # Figures & plots from analysis
├── Models/           # Trained model weights            
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
  author       = {Agam Vuppulury and Gargi Rathi and Srujananjali Medicharla and Vaaruni Desai},
  title        = {The Role of Attention and MLPs in Transformer XOR Computation},
  year         = {2025},
  howpublished = {\url{https://github.com/v-agam/Attention-MLPs-in-Transformer-XOR}}
}

