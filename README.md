# Fibration Symmetry for RL (Forked & Modified from PufferAI)

This is a **fork** of the original [Metta repository](https://github.com/Metta-AI/metta), with custom modifications.

## 🔧 Purpose of the Fork

The goal of this fork is to **extend the functionality** of PufferAI in order to:

> **Enable collapsing of the policy network for certain enviorements using fibration symmetries.**

This enables more efficient training and inference by leveraging symmetries in the game state and action space, thereby reducing redundancy in policy outputs and improving performance.

## 🔄 Differences from Upstream

- Introduced support for **fiber symmetry-based policy collapsing**.
- Modified architecture components to allow group-invariant transformations.
- Custom preprocessing pipeline adjustments to accommodate the symmetry reductions.

In order to use the code for fibration collapsed results, follow the instructions in the file: [Model Collapsing Guide](model_collapsing_guide.MD)


The full documentation of the underlying enviorements is hosted at [puffer.ai](https://huggingface.co/metta-ai/baseline.v0.1.0").  For support -- post here before opening issues.

