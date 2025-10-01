**Torch.fx in KCORE**

For the analysis of fibration symmetries in the agent's computational graph, we use torch.fx. To achieve this, the agent's forward function must use torch.fx-compatible functions (see conditions below). We modified the forward function and the involved scripts to ensure this requirement is met (see details below).


**Torch.fx Compatibility Guide**

Torch.fx is PyTorch's framework for program transformation and capture through symbolic tracing. 

**Compatible Function Types**

**1. Standard PyTorch Operations**
- Basic tensor operations (element-wise, mathematical functions)
- Neural network modules (nn.Linear, nn.Conv2d, etc.)
- Shape and dimension operations (view, reshape, permute)
- Activation functions and normalization layers

**2. Static Control Flow**
- Conditional statements based on static values
- Shape-based decisions (when shapes are known statically)
- Loop structures with predetermined iteration counts
- Multiple return statements from functions

**3. Module-Based Architecture**
- Standard nn.Module subclasses
- Sequential and container modules
- Custom modules with standard PyTorch operations

**Incompatible Patterns**

**1. Dynamic Control Flow**
- Conditional statements based on tensor values
- Loop structures with data-dependent termination
- Runtime-computed branching logic

**2. Non-Traceable Operations**
- In-place tensor modifications
- External library function calls
- Complex Python built-in functions
- Operations that break the computational graph

**3. Dynamic Graph Changes**
- Architecture modifications during forward pass
- Data-dependent graph structure changes
- Runtime module creation and destruction

**Our modifications**

