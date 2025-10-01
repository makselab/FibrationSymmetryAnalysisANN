# Fibration Symmetry for RL (Forked & Modified from Metta)

This is a **fork** of the original [Metta repository](https://github.com/Metta-AI/metta), with custom modifications.

## 🔧 Purpose of the Fork

The goal of this fork is to **extend the functionality** of Metta in order to:

> **Enable collapsing of the policy network for certain enviorements using fibration symmetries.**

This enables more efficient training and inference by leveraging symmetries in the space state and action space, thereby reducing redundancy in policy outputs.

## 🔄 Differences from Upstream

- Introduced support for **fiber symmetry-based policy collapsing**.
- Modified architectural components to allow the use of **torch.fx**.
- Custom preprocessing pipeline adjustments to accommodate the symmetry reductions.

In order to use the code for fibration collapsed results, follow the instructions in the file: [Model Collapsing Guide](model_collapsing_guide.MD)


The full documentation of the underlying enviorements is hosted at [puffer.ai](https://huggingface.co/metta-ai/baseline.v0.1.0").  For support -- post here before opening issues.

