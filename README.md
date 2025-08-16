# How-Transformers-Implement-XOR-A-Mechanistic-Perspective
This repository contains code, experiments, and analysis for studying how small Transformer models learn and compute the XOR function on bit sequences. We extend mechanistic interpretability techniques from the well-studied 4-bit XOR task to the more challenging 7-bit XOR, providing a detailed account of the internal circuits, attention head roles, and neuron selectivity.

🚀 Research Motivation

XOR is a classic benchmark for testing nonlinear computation in neural networks. Despite its simplicity, XOR requires mechanisms beyond linear separability, making it an ideal test case for probing how Transformers actually compute.

Prior work has characterized 4-bit XOR, showing a division of labor between attention (copying inputs) and MLPs (introducing nonlinearity).

Here, we investigate whether these mechanisms scale to 7-bit XOR, where the model must generalize over longer offsets.

🧪 Experiments

We trained a 2-layer, 2-head Transformer on the 7-bit XOR task and analyzed it using mechanistic interpretability tools:

Attention Analysis

Head 0 emerges as the primary routing head, attending from positions 7–13 back to 0–6.

Path patching and ablations confirm that this head is causally necessary (Head 0 ablation → acc ≈ 0.379).

Head 1 plays a secondary/auxiliary role (ablation → acc ≈ 0.838).

Linear Probes

Early layers encode only local bits (bj).

Attention injects the far bit (bi), but XOR remains inseparable.

At the MLP layer, XOR becomes linearly separable (probe acc = 1.0).

Final residual streams mix both raw inputs and the XOR feature.

Neuron Selectivity

MLP neurons specialize into 00, 01, 10, 11 detectors.

XOR is reconstructed as a linear combination of pro-XOR (01,10) and anti-XOR (00,11) neuron groups.

Compared to 4-bit XOR, the clustering is less sharp, suggesting more distributed representations in 7-bit.

Ablation Studies

Random-K vs. Top-K ablations show that a small subset of neurons (≈16 out of 256) suffices for XOR.

Accuracy remains robust until ≈128 neurons are removed, then collapses.

Overlap-based rankings fail, confirming XOR is not aligned to a single direction but distributed across subspaces.

📊 Key Findings

Stable Mechanism Across Tasks: The attention → MLP → XOR pipeline discovered in 4-bit tasks persists in the 7-bit model.

Head Assignment is Flexible: In 4-bit, Head 1 was the router; in 7-bit, Head 0 takes that role. This indicates head specialization is determined by training dynamics, not architecture.

Neuron Efficiency: XOR computation is highly redundant yet compressible — only a handful of neurons are strictly necessary.

Scalable Circuit Motif: Despite longer offsets, the Transformer reuses the same compositional circuit structure.
