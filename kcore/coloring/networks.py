import copy

import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list

from coloring.nn_modules import fibration_for_nn_module
from coloring.functions_methods import fibration_for_function
from coloring.collapse_nn import collapse_nn_module

# ------------------------------------------------------------------------
# Color Graph Propagation Runner
# ------------------------------------------------------------------------

class ColorPropagator:
    """
    A class for propagating color information through a Pytorch model.
    
    This class uses torch.fx to trace the model and perform color propagation
    through the computational graph, enabling analysis of symmetries in neural networks.
    
    Attributes:
        model: The original PyTorch model
        traced: Symbolically traced version of the model using torch.fx
        modules: Dictionary of named modules in the traced model
        parameters: Dictionary storing parameter tensors encountered during propagation
        colors: Dictionary storing color information for each node in the graph
    """

    def __init__(self, model):
        """
        Initialize the ColorPropagator with a Pytorch model.
        
        Args:
            model: A PyTorch model to be analyzed.
        """

        self.model = model
        self.traced = torch.fx.symbolic_trace(model)
        self.modules = dict(self.traced.named_modules())
        self.parameters = {}
        self.colors = {}

    def coloring(self, dict_input_tensors, threshold, origin_node):
        """
        Perform color propagation through the model graph starting from a specified node.
        
        This method propagates color information through the computational graph.
        
        Args:
            dict_input_tensors: Dictionary mapping input names to tensor values
            threshold: Threshold value for fibration calculations. Range (0-2).
                        0: 
                        2: Full collapse
            origin_node: The starting node in the graph for color propagation
            
        Returns:
            Dictionary containing color information for all processed nodes


        Implementation details: 
        - Traces through the computational graph using torch.fx
        - Processes different types of operations (placeholder, get_attr, call_function, call_method, call_module)
        - Analyze the fibrations symemtries.
        """

        # DEBUG
        # print('====================================================================')
        # print(self.model)
        # print('Generated Graph')

        # for node in self.traced.graph.nodes:
        #     print(node.name, node.args, node.target, node.op)
        # print('====================================================================')


        found_origin = False

        for node in self.traced.graph.nodes:
            # Node processing logic handles different operation types:
            # - Placeholder nodes: Input tensors
            # - Get_attr nodes: Model parameters  
            # - Call_function/method nodes: Function operations
            # - Call_module nodes: Neural network modules
            
            # Color propagation continues until the entire graph is processed or an error occurs

            if not found_origin:
                if node.name == "output":
                    self.outputs = self._get_output_name(node.args)

                elif node.op == 'placeholder':
                    input_name = node.name
                    input_net = dict_input_tensors[input_name]
                    self._get_input_colors(input_name,input_net, self.colors)
                    # print('Input:', node.name, self.colors[input_name])
                    continue

                elif node.op == 'get_attr':
                    parameter_name = node.name
                    parameter_tensor = self._get_attr(self.traced, node.target).data
                    self.parameters[parameter_name] = parameter_tensor
                    # print('Parameter:', node.name, parameter_tensor)
                    continue

                elif node.name == origin_node:
                    found_origin = True
                    
                    for a in node.args:
                        a_name = a.name
                        if a_name not in self.colors:
                            self._get_input_colors(a_name,dict_input_tensors[a_name], self.colors)


            if found_origin:
                if node.name == "output":
                    print(node.name, node.args, node.target, node.op)
                    self.outputs = [valor for tupla in node.args for valor in tupla]
                    self.outputs = self._get_output_name(node.args)

                elif node.op == 'get_attr':
                    # print('Parameter:', node.name)
                    parameter_name = node.name
                    parameter_tensor = self._get_attr(self.traced, node.target).data
                    self.parameters[parameter_name] = parameter_tensor
                    continue

                arg_colors    = []
                arg_param     = []
                arg_other     = []


                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        arg_name = arg.name
                        if arg_name in self.parameters:
                            arg_param.append(self.parameters[arg_name])
                        else:
                            arg_colors.append(self.colors[arg.name])
                    elif isinstance(arg, immutable_list):
                        arg_colors.append([self.colors[ee.name] for ee in arg])            
                    else:
                        arg_other.append(arg)

                # DEBUG: print(node.name, node.op, node.target, node.args, arg_colors, arg_param, arg_other)

                if node.op in ['call_function', 'call_method']:

                    if len(arg_colors) == 0:
                        node_name = node.name
                        args_transformation = tuple(arg_param + arg_other)

                        if 'method' in node.op:
                            method_name = node.target
                            method = getattr(args_transformation[0], method_name)
                            new_value = method(*args_transformation[1:], **node.kwargs)
                        else:
                            function = node.target
                            new_value = function(*args_transformation, **node.kwargs)
                        
                        self.parameters[node_name] = new_value
                        continue

                    else:
                        function_name = node.name
                        function = node.target
                        out_colors = fibration_for_function(arg_colors, arg_param, arg_other, threshold, function)

                        self.colors[function_name] = out_colors

                if node.op == 'call_module':
                    module_name = node.name
                    module = self.modules[node.target]
                    module.ID = module_name 
                    try:
                        # print(module_name)
                        out_colors = fibration_for_nn_module(arg_colors, threshold, module)
                        self.colors[module_name] = out_colors
                    except NotImplementedError as e:
                        print(f"Stopping coloring propagation at node {node.name}: {str(e)}")
                        return self.colors
                    except Exception as e:
                        print(f"Stopping coloring propagation at node {node.name}: {str(e)}")
                        return self.colors

                # print('Output:', out_colors)

        #     # print('-------------------------------')
        # print(self.colors)
        # print(self.parameters)

        # print('Coloring Done')

        # print('====================================================================')

        return out_colors

    def collapse_model(self):
        """
        Create a collapsed version of the model based on color propagation analysis.
        
        Returns:
            A new model instance with certain modules collapsed based on
            the color propagation results. Modules containing 'actor', 'value_fn',
            or 'cnn2' in their names are preserved from collapsing.

        Implementations:
        # Creates a deep copy of the original model
        # Iterates through modules and applies collapse transformation
        """

        base_model = copy.deepcopy(self.model)

        named_modules = dict(base_model.named_modules())

        for node in self.traced.graph.nodes:
            if node.op == 'call_module': 

                node_target = node.target

                collapse_out = True
                if 'actor' in node_target or 'value_fn' in node_target or 'cnn2' in node_target:
                    collapse_out = False 

                collapse_in = True

                original_module = named_modules[node_target]
                collapsed_module = collapse_nn_module(original_module, collapse_in=collapse_in, collapse_out=collapse_out)

                path_components = node.target.split('.')
                parent_module = base_model

                for component in path_components[:-1]:
                    parent_module = getattr(parent_module, component)

                setattr(parent_module, path_components[-1], collapsed_module)

        return base_model

    def _get_attr(self, module, target):
        """
        Recursively get attribute/parameter from module using dot-separated target path.
        
        Args:
            module: The module to start attribute/parameter lookup from
            target: Dot-separated string specifying the attribute path
            
        Returns:
            The requested attribute/parameter value

        Traverses the module hierarchy using the dot-separated path
        Returns the final attribute value
        """

        attr = module
        for name in target.split('.'):
            attr = getattr(attr, name)
        return attr

    def _count_input_features(self, tensor_sample):
        """
        Calculate the number of input features based on tensor dimensions.
        
        Args:
            tensor_sample: A sample tensor from which to determine feature count
            
        Returns:
            Integer representing the number of input features, or None if
            the dimension pattern is not recognized

        Handles different tensor dimensionalities:
        - 1D tensors: vector inputs
        - 2D tensors: sequence data (seq_len, features)  
        - 3D tensors: image-like data (channels, height, width)
        Returns None for unrecognized dimensions
        """

        if tensor_sample.dim() == 1:
            return tensor_sample.shape[0]  # vector input
        elif tensor_sample.dim() == 2:
            return tensor_sample.shape[-1]  # (seq_len, D)
        elif tensor_sample.dim() == 3:
            return tensor_sample.shape[0]  # likely (channels, height, width)
        else:
            print('Check dimensions')
            return None

    def _get_input_colors(self,input_name, x, dictionary_colors):
        """
        Initialize color information for input tensors.
        
        Args:
            input_name: Name identifier for the input
            x: Input data (tensor, dictionary, or other structure)
            dictionary_colors: Dictionary to store the color mappings

        Handles different input types:
        - Tensor inputs: creates color range based on feature count
        - Dictionary inputs: recursively processes each key-value pair
        - None inputs: stores None in color dictionary
        - Dataclass-like inputs: processes each field recursively

        """

        if type(x) == torch.Tensor:
            num_input_colors = self._count_input_features(x[0]) # check generalization
            input_colors = torch.arange(num_input_colors)
            dictionary_colors[input_name] = input_colors
        
        elif x is None:
            dictionary_colors[input_name] = None

        elif type(x) == dict:
            dictionary_colors[input_name] = {}

            for f in x.keys():
                self._get_input_colors(f, x[f], dictionary_colors[input_name])            
        else:
            _fields = [f.name for f in x.fields()]
            dictionary_colors[input_name] = {}

            for f in _fields:
                self._get_input_colors(f, getattr(x, f), dictionary_colors[input_name])

    def _get_output_name(self, outputs):
        """
        Extract names from the graph output nodes.
        
        Args:
            outputs: Graph output nodes
            
        Returns:
            List of output names, with None values preserved

        Flattens the output structure into a simple list
        Preserves None values in the output list
        """

        list_ = []

        for arg in outputs:
            if arg is None:
                list_.append(None)
            else:
                for valor in tupla:
                    list_.append(valor)