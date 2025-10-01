import torch
# from sklearn.cluster import AgglomerativeClustering
from torch import zeros, mm, cat, tensor, where, unique, repeat_interleave, vstack, eye
from torch.nn.functional import normalize

def fibration_for_function(input_colors, parameters, other, threshold, operation):

    if "matmul" in str(operation):
        if len(parameters)==1:
            in_colors =  input_colors[0]
            weights = parameters[0]
            weights = weights.T if (len(input_colors[0]) == weights.shape[0]) else weights
            dim_out, dim_in  = weights.shape

            if dim_out == 1:
                return tensor([0])

            idx_in_colors  = unique(in_colors)
            num_in_colors  = len(idx_in_colors)
            collapse_weights = zeros((dim_out, num_in_colors))

            for pos_color, color in enumerate(idx_in_colors): 
                indices_k = where(in_colors == color)[0]
                collapse_weights[:, pos_color] = weights[:, indices_k].sum(axis=1)

            collapse_weights_norm = normalize(collapse_weights, dim=1)
            distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
            distance = distance.cpu().numpy()

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                linkage='average',
                metric='precomputed')

            clusters = tensor(clustering.fit_predict(distance))

            return clusters

    if "add" in str(operation) or "sub" in str(operation):
        if len(parameters)==1:
            in_colors =  input_colors[0]
            dim_out = in_colors.shape[0]
            weights = eye(dim_out)
            bias = parameters[0]
            if "sub" in str(operation): bias = -bias

            if dim_out == 1:
                return tensor([0])

            idx_in_colors  = unique(in_colors)
            num_in_colors  = len(idx_in_colors)
            collapse_weights = zeros((dim_out, num_in_colors))

            for pos_color, color in enumerate(idx_in_colors): 
                indices_k = where(in_colors == color)[0]
                collapse_weights[:, pos_color] = weights[:, indices_k].sum(axis=1)

            dev_ = collapse_weights.device
            collapse_weights = cat((collapse_weights, bias.to(dev_).unsqueeze(1)), dim=1)

            collapse_weights_norm = normalize(collapse_weights, dim=1)
            distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
            distance = distance.cpu().numpy()

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                linkage='average',
                metric='precomputed')

            clusters = tensor(clustering.fit_predict(distance))

            return clusters

        elif len(other)==1:
            return input_colors[0]

        else:
            mtx = torch.stack(input_colors, dim=0).T
            _, clusters = torch.unique(mtx, dim=0, return_inverse=True)    
            return clusters

    if "mul" in str(operation):
        input_colors_wo_none = [x for x in input_colors if x is not None]
        
        if len(parameters)==1:
            print('implement')
        elif len(other)==1:
            print('implement')
        else:
            mtx = torch.stack(input_colors_wo_none, dim=0).T
            _, clusters = torch.unique(mtx, dim=0, return_inverse=True)    
            return clusters

    if "dim" == str(operation):
        return tensor([0])

    if "getattr" in str(operation):
        if other[0] == 'shape':
            return None
            # return tensor([0]*100)

        if other[0] == 'device':
            return None

        if isinstance(input_colors[0], dict) and other[0] in input_colors[0]:
            return input_colors[0][other[0]]

    if "getitem" in str(operation):
        var = input_colors[0]

        if var is None:
            return None

        if type(var) == dict:
            return var[other[0]]

        if len(other) == 0:
            return var

        # Valid for Metta (len(other) == dim(input) para que tenga efecto)
        if len(other) == 1:
            return var

        selection = other[0]

        if type(selection) == int and type(var) != tuple: selection = [selection]

        return var[selection]
        
    if "eq" in str(operation) or "function ne" in str(operation) or "ge" in str(operation) or "le" in str(operation):
        return None

    if "and" in str(operation) or "assert" in str(operation):
        return None

    if "reshape" == str(operation) or 'transpose' == str(operation):
        return input_colors[0]

    if "float" in str(operation):
        return input_colors[0]

    # if "cat" in str(operation):
    #     return torch.cat(flat_inputs)

    # if "size" == str(operation):
    #     return input_colors_list[0]

    # if "flatten" == str(operation):
    #     return torch.repeat_interleave(input_colors_list[0],64) #check

    if "truediv" in str(operation) or "floordiv" in str(operation):
        if len(input_colors)==2:
            mtx = torch.stack(input_colors, dim=0).T
            _, clusters = torch.unique(mtx, dim=0, return_inverse=True)    
            return clusters

        else:
            return input_colors[0]

    if "to" in str(operation):
        return input_colors[0]

    if "cat" in str(operation):
        return torch.cat(input_colors[0])

    if "rshift" in str(operation):
        return input_colors[0]

    if "long" in str(operation):
        return input_colors[0]

    if "new_zeros" == str(operation):
        return tensor([0])

    if "unsqueeze" == str(operation):
        return tensor([0])

    if "view" == str(operation):
        return input_colors[0]

    if "detach" == str(operation):
        return input_colors[0]

    if "set" == str(operation):
        input_colors[0][other[0]] = input_colors[1]
        return input_colors[1]

    if "numel" == str(operation):
        return input_colors[0]

    else:
        print('Error')
        exit()