import torch.nn as nn
from torch.nn.functional import normalize
from torch import zeros, mm, cat, tensor, where, unique, repeat_interleave, vstack, arange, int32, int64
import torch

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from numpy import fill_diagonal

# ====================================================================

def fibration_for_nn_module(input_colors, threshold, module):
    if isinstance(module, nn.Conv2d):
        return fibration_conv2d(input_colors, threshold['cnn'], module)
    elif isinstance(module, nn.Linear):
        if 'critic' in module.ID:
            return fibration_linear(input_colors, threshold['critic'], module)
        else:
            return fibration_linear(input_colors, threshold['linear'], module)
    elif isinstance(module, nn.LSTM):
        return fibration_lstm(input_colors, threshold['lstm'], module)
    elif isinstance(module, nn.ReLU):
        return fibration_relu(input_colors, threshold, module)
    elif isinstance(module, nn.Tanh):
        return fibration_tanh(input_colors, threshold, module)
    elif isinstance(module, nn.Flatten):
        return fibration_flatten(input_colors, threshold, module)
    elif isinstance(module, nn.BatchNorm2d):
        return fibration_normalization(module, input_colors, threshold)
    elif isinstance(module, nn.Embedding):
        raise Exception(
            f"Embedding stops the coloring from spreading."
        )
    else:
        # Módulo no soportado - detener la ejecución
        raise NotImplementedError(
            f"Module type {type(module)} is not supported in fibration. "
            f"Coloring propagation will stop. "
        )
    # else:
    #     print('break')
    #     return tensor([0])
    #     # module_prop = ColorPropagator(module)
    #     # num_f = len(in_colors)
    #     # dummy_input = torch.randn(1, num_f)
    #     # prop.run({'x': dummy_input} ,threshold=threshold)

# ====================================================================

def fibration_linear(input_colors, threshold, module):
    in_colors =  input_colors[0]
    weights = module.weight.data
    bias = module.bias.data

    dim_out, dim_in  = weights.shape
    module.in_colors = input_colors

    if dim_out == 1:
        clusters = tensor([0])
        module.out_colors = clusters
        return clusters

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
    distance = torch.clamp(distance, min=0.0)
    distance = distance.cpu().numpy()
    fill_diagonal(distance, 0)
    distance = 0.5 * (distance + distance.T)
    distance_condensed = squareform(distance, force='tovector')

    Z = linkage(distance_condensed, method='average')
    clusters = tensor(fcluster(Z, t=threshold, criterion='distance'),dtype=int64)-1

    module.out_colors = clusters

    return clusters

# ====================================================================

def fibration_conv2d(input_colors, threshold, module):
    in_colors =  input_colors[0]
    weights = module.weight.data
    bias = module.bias.data

    out_n, in_n, hx, hy = weights.shape
    weights = weights.view(out_n, in_n, -1)

    idx_in_colors = unique(in_colors)
    num_in_colors  = len(idx_in_colors)
    collapse_weights = zeros((out_n, num_in_colors, hx*hy))

    for pos_color, color in enumerate(idx_in_colors): 
        indices_k = where(in_colors == color)[0]
        collapse_weights[:, pos_color, :] = weights[:, indices_k, :].sum(axis=1)

    collapse_weights = collapse_weights.view(out_n,-1)

    dev_ = collapse_weights.device
    collapse_weights = cat((collapse_weights, bias.to(dev_).unsqueeze(1)), dim=1)

    collapse_weights_norm = normalize(collapse_weights, dim=1)
    distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
    distance = torch.clamp(distance, min=0.0)
    distance = distance.cpu().numpy()
    fill_diagonal(distance, 0)
    distance_condensed = squareform(distance, force='tovector')

    Z = linkage(distance_condensed, method='average')
    clusters = tensor(fcluster(Z, t=threshold, criterion='distance'),dtype=int64)-1

    module.in_colors = input_colors
    module.out_colors = clusters

    return clusters

# ====================================================================

def fibration_lstm(input_colors, threshold, module):

    num_layers = module.num_layers
    T=5
    in_colors = input_colors[0]
    module.in_colors = input_colors
    module.out_colors = []

    for layer_idx in range(1):
        weight_ih = getattr(module, f'weight_ih_l{layer_idx}').data
        weight_hh = getattr(module, f'weight_hh_l{layer_idx}').data
        bias_ih = getattr(module, f'bias_ih_l{layer_idx}').data
        bias_hh = getattr(module, f'bias_hh_l{layer_idx}').data

        dim_h, dim_in = weight_ih.shape
        h_clusters = zeros(dim_h//4, dtype=int32)
        c_clusters = zeros(dim_h//4, dtype=int32)

        for t in range(T):

            # (1) Collapse W_ii,...W_io based on in_cluster.
            idx_in_colors  = unique(in_colors)
            num_in_colors = len(idx_in_colors)
            collapse_weights_i = zeros((dim_h, num_in_colors))

            for pos_color, color in enumerate(idx_in_colors): 
                indices_k = where(in_colors == color)[0]
                collapse_weights_i[:, pos_color] = weight_ih[:, indices_k].sum(axis=1)

            # (2) Collapse W_hi,...W_ho based on h_cluster.

            idx_h_clusters  = unique(h_clusters)
            num_h_clusters = len(idx_h_clusters)
            collapse_weights_h = zeros((dim_h, num_h_clusters))

            for pos_color, color in enumerate(idx_h_clusters): 
                indices_k = where(h_clusters == color)[0]
                collapse_weights_h[:, pos_color] = weight_hh[:, indices_k].sum(axis=1)

            collapse_weights = cat((collapse_weights_i,collapse_weights_h),dim=1)

            # (3) Clusters of the gates

            collapse_weights_norm = normalize(collapse_weights, dim=1)
            distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
            distance = torch.clamp(distance, min=0.0)
            distance = distance.cpu().numpy()
            fill_diagonal(distance, 0)
            distance_condensed = squareform(distance, force='tovector')

            Z = linkage(distance_condensed, method='average')
            gates_clusters = tensor(fcluster(Z, t=threshold, criterion='distance'))

            # (4) (F,C,I,G) clusters
            gates_cluster_matrix = gates_clusters.reshape(4,dim_h//4)

            # (5) c_clusters
            new_c_clusters = vstack((gates_cluster_matrix[:3,:], c_clusters))
            _, c_clusters = unique(new_c_clusters.T, dim=0, return_inverse=True)

            # (6) h_clusters
            new_h_clusters = vstack((gates_cluster_matrix[3,:], c_clusters))
            _, h_clusters = unique(new_h_clusters.T, dim=0, return_inverse=True)

        # in_colors= h_clusters
        

    for layer_idx in range(num_layers):
        module.out_colors.append(h_clusters)

    return h_clusters

# ====================================================================

def fibration_relu(input_colors, threshold, module):
    return input_colors[0]

def fibration_tanh(input_colors, threshold, module):
    return input_colors[0]
# ====================================================================

def fibration_normalization(layer, in_clusters, threshold):
    return in_clusters

# ====================================================================

def fibration_embedding(layer, in_clusters, threshold):
    weights = layer.weight.data
    weights_norm = normalize(weights, dim=0)
    distance = 1 - mm(weights_norm.T, weights_norm)
    distance = distance.cpu().numpy()

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage='average',
        metric='precomputed')

    clusters = tensor(clustering.fit_predict(distance))

    layer.in_colors = in_clusters
    layer.out_colors = clusters

    return clusters

# ====================================================================

def fibration_flatten(input_colors, threshold, module, times=54):
    # Puffeflib 54, Metta 1
    times = 1
    num_colors = len(input_colors[0])*times

    return arange(num_colors)