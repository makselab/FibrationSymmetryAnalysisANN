# Fibration Symmetry Analysis (Forked & Modified from Metta)

This is a **fork** of the original [Metta repository](https://github.com/Metta-AI/metta).

## 🔧 Purpose of the Fork

The goal of this fork is to present a generic fibration symmetry analysis to collapse generic neural network models. The code is tested and validated in Metta agents.

## 🔄 Differences from Upstream

- Introduced support and documentation for fibration symmetry-based analysis and collapsing.
- Modified architectural components to allow the use of torch.fx.

In order to use the code for fibration symmetry analysis:
- in Metta Agents (incluiding test and validation), follow the instructions in the file: [Metta Agent Collapsing Guide](metta_collapsing_guide.MD)
- in any generic neural network model, follow the instruction in the file: [Generic Model Collapsing Guide](generic_collapsing_guide.MD) 


The full documentation of the underlying enviorements is hosted at [Metta](https://huggingface.co/metta-ai/baseline.v0.1.0").  

For support -- post here before opening issues.