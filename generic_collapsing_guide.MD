FIBRATION SYMMETRIES IN GENERIC MODELS: STEP-BY-STEP GUIDE
=================================================

Use the class 'ColorPropagator' defined in ./kcore/coloring/networks.py to:

1. obtain colors of the computational graph of the neural network model
2. collapse the neural network model

First, set up attribute "_ traced" in the modules in the model:

    for kk, cc in model.items():
        cc._traced=False

Then, initializate the object 'propagator'

    propagator= ColorPropagator(model)

1.RUN COLORING
-----------------------------

Use the method '.coloring' to obtain the colors:

    propagator.coloring(inputs_net,
                threshold={'linear':linear_thr, 
                            'cnn': cnn_thr, 
                            'lstm':lstm_thr,
                            'critic':critic_thr}, 
                origin_node='name_node')

where 
- inputs_net: Random tensor with the same dimensions of the inputs.
- threshold: Thresholds for the fibration calculation.

2.RUN COLLAPSE
-----------------------------
    base_model = propagator.collapse_model()
    num_params_base = sum(p.numel() for p in base_model.parameters())
    per_reduction = num_params_base/num_params