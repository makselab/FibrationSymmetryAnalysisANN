import torch.nn as nn
import torch
from torch.nn.functional import normalize
from torch import zeros, mm, cat, tensor, where, unique, repeat_interleave, vstack

# ====================================================================

def collapse_nn_module(module, collapse_in, collapse_out):
    if isinstance(module, nn.Conv2d):
        return collapse_conv2d(module, collapse_in, collapse_out)
    elif isinstance(module, nn.Linear):
        return collapse_linear(module, collapse_in, collapse_out)
    elif isinstance(module, nn.LSTM):
        return collapse_lstm(module, collapse_in, collapse_out)
    else:
        return module

# ====================================================================

def collapse_linear(module, collapse_in=True, collapse_out=True):
    W = module.weight.data
    b = module.bias.data
    dim_out, dim_in = W.shape
    dev = module.weight.device

    if collapse_in:
        num_in_colors = torch.unique(module.in_colors[0]).shape[0]
        in_mtx_partition   = torch.zeros(num_in_colors, dim_in).scatter_(0, module.in_colors[0].unsqueeze(0), 1).to(dev)
        W_coll = W @ in_mtx_partition.T 
    else:
        num_in_colors = dim_in
        W_coll = W 

    if collapse_out:
        num_out_colors = torch.unique(module.out_colors).shape[0]
        out_mtx_partition  = torch.zeros(num_out_colors, dim_out).scatter_(0, module.out_colors.unsqueeze(0), 1).to(dev)
        out_sizes_clusters = torch.mm(out_mtx_partition, torch.ones(dim_out, 1).to(dev))
        W_coll = (out_mtx_partition @ W_coll) / out_sizes_clusters.view(-1, 1)
        b_coll = out_mtx_partition @ b / out_sizes_clusters.view(-1,)
    else: 
        num_out_colors = dim_out
        b_coll = b

    module_coll = nn.Linear(num_in_colors, num_out_colors, device=dev)
    module_coll.weight.data = W_coll
    module_coll.bias.data = b_coll

    return module_coll

# ====================================================================

def collapse_conv2d(module, collapse_in=True, collapse_out=True):
    W = module.weight.data
    b = module.bias.data
    dim_out, dim_in, _, _ = W.shape
    dev = module.weight.device

    if collapse_in:
        num_in_colors = torch.unique(module.in_colors[0]).shape[0]
        in_mtx_partition   = torch.zeros(num_in_colors, dim_in).scatter_(0, module.in_colors[0].unsqueeze(0), 1).to(dev)
        W_resh = W.permute(0, 2, 3, 1)
        W_coll = W_resh @ in_mtx_partition.T
        W_coll = W_coll.permute(0, 3, 1, 2)
    else:
        num_in_colors = dim_in
        W_coll = W 

    if collapse_out:
        num_out_colors = torch.unique(module.out_colors).shape[0]
        out_mtx_partition  = torch.zeros(num_out_colors, dim_out).scatter_(0, module.out_colors.unsqueeze(0), 1).to(dev)
        out_sizes_clusters = torch.mm(out_mtx_partition, torch.ones(dim_out, 1).to(dev))
        W_coll = torch.einsum('ai,ibxy->abxy', out_mtx_partition, W_coll)
        W_coll = W_coll / out_sizes_clusters.view(-1, 1, 1, 1)
        b_coll = out_mtx_partition @ b / out_sizes_clusters.view(-1,)
    else: 
        num_out_colors = dim_out
        b_coll = b 

    module_coll = nn.Conv2d(
        in_channels=num_in_colors,
        out_channels=num_out_colors,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode, device=dev)

    module_coll.weight.data = W_coll
    module_coll.bias.data = b_coll

    return module_coll

# ====================================================================

def collapse_lstm(module, collapse_in=True, collapse_out=True):

    num_layers = module.num_layers
    in_colors = module.in_colors[0]
    dict_w = [None for layer_idx in range(num_layers)]
    dict_bias = [None for layer_idx in range(num_layers)]

    matrix_in_partition = [None for layer_idx in range(num_layers)]
    matrix_h_partition = [None for layer_idx in range(num_layers)]
    sizes_h_clusters = [None for layer_idx in range(num_layers)]

    for layer_idx in range(num_layers):
        # ==============================================

        h_colors = module.out_colors[layer_idx]
        in_dim = len(in_colors)
        h_dim = len(h_colors)

        # ==============================================

        w_ih = getattr(module, f'weight_ih_l{layer_idx}').data
        w_hh = getattr(module, f'weight_hh_l{layer_idx}').data        
        b_ih = getattr(module, f'bias_ih_l{layer_idx}').data
        b_hh = getattr(module, f'bias_hh_l{layer_idx}').data

        dev = w_ih.device

        Whi, Whf, Whg, Who = w_hh[:h_dim,:], w_hh[h_dim:2*h_dim,:], w_hh[2*h_dim:3*h_dim,:], w_hh[3*h_dim:,:]
        Wii, Wif, Wig, Wio = w_ih[:h_dim,:], w_ih[h_dim:2*h_dim,:], w_ih[2*h_dim:3*h_dim,:], w_ih[3*h_dim:,:]

        bhi, bhf, bhg, bho = b_hh[:h_dim], b_hh[h_dim:2*h_dim], b_hh[2*h_dim:3*h_dim], b_hh[3*h_dim:]
        bii, bif, big, bio = b_ih[:h_dim], b_ih[h_dim:2*h_dim], b_ih[2*h_dim:3*h_dim], b_ih[3*h_dim:]

        dict_w[layer_idx] = {'hi': Whi, 'hf': Whf, 'hg': Whg, 'ho': Who,
                    'ii': Wii, 'if': Wif, 'ig': Wig, 'io':Wio}

        dict_bias[layer_idx] = {'hi': bhi, 'hf': bhf, 'hg': bhg, 'ho': bho,
                'ii': bii, 'if': bif, 'ig': big, 'io':bio}

        # ==============================================

        num_in_clusters = torch.unique(in_colors).shape[0]
        matrix_in_partition_ll = torch.zeros(num_in_clusters, in_dim)
        matrix_in_partition_ll.scatter_(0, in_colors.unsqueeze(0), 1)
        # sizes_in_clusters = torch.mm(matrix_in_partition, torch.ones(in_dim, 1))
        matrix_in_partition[layer_idx] = matrix_in_partition_ll.to(dev)

        num_h_clusters = torch.unique(h_colors).shape[0]
        matrix_h_partition_ll = torch.zeros(num_h_clusters, h_dim)
        matrix_h_partition_ll.scatter_(0, h_colors.unsqueeze(0), 1)
        sizes_h_clusters_ll = torch.mm(matrix_h_partition_ll, torch.ones(h_dim, 1))

        matrix_h_partition[layer_idx] = matrix_h_partition_ll.to(dev)
        sizes_h_clusters[layer_idx] = sizes_h_clusters_ll.to(dev)


        # ==============================================

        for name, W in dict_w[layer_idx].items():
            W_collap = torch.mm(matrix_h_partition[layer_idx], W) / sizes_h_clusters[layer_idx].view(-1, 1)

            partition = matrix_h_partition[layer_idx] if 'h' in name else matrix_in_partition[layer_idx]
            dict_w[layer_idx][name] = torch.mm(W_collap, partition.T)

        for name, b in dict_bias[layer_idx].items():
            bias_collap = matrix_h_partition[layer_idx] @ b / sizes_h_clusters[layer_idx]
            dict_bias[layer_idx][name] = bias_collap[0, :]

        # ==============================================

        in_colors = h_colors


    # Copy of the agent
    module_coll = nn.LSTM(torch.unique(module.in_colors[0]).shape[0],
                        torch.unique(module.out_colors[0]).shape[0], 
                        num_layers=num_layers)

    with torch.no_grad():
        for layer_idx in range(num_layers):
            dd = dict_w[layer_idx]
            bb = dict_bias[layer_idx]


            setattr(module_coll, f'weight_ih_l{layer_idx}', nn.Parameter(torch.cat([dd['ii'], dd['if'], dd['ig'], dd['io']]), requires_grad=True))
            setattr(module_coll, f'weight_hh_l{layer_idx}', nn.Parameter(torch.cat([dd['hi'], dd['hf'], dd['hg'], dd['ho']]), requires_grad=True))

            setattr(module_coll, f'bias_ih_l{layer_idx}', nn.Parameter(torch.cat([bb['ii'], bb['if'], bb['ig'], bb['io']]), requires_grad=True))
            setattr(module_coll, f'bias_hh_l{layer_idx}', nn.Parameter(torch.cat([bb['hi'], bb['hf'], bb['hg'], bb['ho']]), requires_grad=True))

    # print(module_coll)

    # for nam, pp in module_coll.named_parameters():
    #     print(nam,pp.shape)

    # exit()

    return module_coll

    # in_dim = module.input_size
    # h_dim = module.hidden_size

    # # Layer 1
    # Whi, Whf, Whg, Who = module.weight_hh_l0.data[:h_dim,:], module.weight_hh_l0.data[h_dim:2*h_dim,:], module.weight_hh_l0.data[2*h_dim:3*h_dim,:], module.weight_hh_l0.data[3*h_dim:,:]
    # Wii, Wif, Wig, Wio = module.weight_ih_l0.data[:h_dim,:], module.weight_ih_l0.data[h_dim:2*h_dim,:], module.weight_ih_l0.data[2*h_dim:3*h_dim,:], module.weight_ih_l0.data[3*h_dim:,:]

    # bhi, bhf, bhg, bho = module.bias_hh_l0.data[:h_dim], module.bias_hh_l0.data[h_dim:2*h_dim], module.bias_hh_l0.data[2*h_dim:3*h_dim], module.bias_hh_l0.data[3*h_dim:]
    # bii, bif, big, bio = module.bias_ih_l0.data[:h_dim], module.bias_ih_l0.data[h_dim:2*h_dim], module.bias_ih_l0.data[2*h_dim:3*h_dim], module.bias_ih_l0.data[3*h_dim:]

    # # Layer 2
    # Whhl1 = module.weight_hh_l1.data
    # Wihl1 = module.weight_ih_l1.data
    # bhhl1 = module.bias_hh_l1.data
    # bihl1 = module.bias_ih_l1.data

    # # Num of clusters, Matrix of partition and sizes of the clusters
    # num_in_clusters = torch.unique(module.in_colors[0]).shape[0]
    # matrix_in_partition = torch.zeros(num_in_clusters, in_dim)
    # matrix_in_partition.scatter_(0, module.in_colors[0].unsqueeze(0), 1)
    # sizes_in_clusters = torch.mm(matrix_in_partition, torch.ones(in_dim, 1))

    # num_h_clusters = h_dim
    # matrix_h_partition = torch.eye(num_h_clusters)
    # # num_h_clusters = torch.unique(module.out_colors).shape[0]
    # # matrix_h_partition = torch.zeros(num_h_clusters, h_dim)
    # # matrix_h_partition.scatter_(0, module.out_colors.unsqueeze(0), 1)
    # sizes_h_clusters = torch.mm(matrix_h_partition, torch.ones(h_dim, 1))

    # dict_w = {'hi': Whi, 'hf': Whf, 'hg': Whg, 'ho': Who,
    #             'ii': Wii, 'if': Wif, 'ig': Wig, 'io':Wio}

    # for name, W in dict_w.items():
    #     W_collap = torch.mm(matrix_h_partition, W) / sizes_h_clusters.view(-1, 1)

    #     partition = matrix_h_partition if 'h' in name else matrix_in_partition
    #     dict_w[name] = torch.mm(W_collap, partition.T)

    # weight_hh_l0_collap = torch.cat([dict_w['hi'], dict_w['hf'], dict_w['hg'], dict_w['ho']])
    # weight_ih_l0_collap = torch.cat([dict_w['ii'], dict_w['if'], dict_w['ig'], dict_w['io']])
    
    # dict_bias = {'hi': bhi, 'hf': bhf, 'hg': bhg, 'ho': bho,
    #         'ii': bii, 'if': bif, 'ig': big, 'io':bio}
    
    # for name, b in dict_bias.items():
    #     bias_collap = matrix_h_partition @ b / sizes_h_clusters
    #     dict_bias[name] = bias_collap[0, :]

    # bias_hh_l0_collap = torch.cat([dict_bias['hi'], dict_bias['hf'], dict_bias['hg'], dict_bias['ho']])
    # bias_ih_l0_collap = torch.cat([dict_bias['ii'], dict_bias['if'], dict_bias['ig'], dict_bias['io']])

    # # Copy of the agent
    # module_coll = nn.LSTM(num_in_clusters,num_h_clusters, num_layers=2)
    # module_coll.weight_hh_l0.data = weight_hh_l0_collap
    # module_coll.weight_ih_l0.data = weight_ih_l0_collap
    # module_coll.bias_hh_l0.data = bias_hh_l0_collap
    # module_coll.bias_ih_l0.data = bias_ih_l0_collap

    # module_coll.weight_hh_l1.data = module.weight_hh_l1.data
    # module_coll.weight_ih_l1.data = module.weight_ih_l1.data
    # module_coll.bias_hh_l1.data = module.bias_hh_l1.data
    # module_coll.bias_ih_l1.data = module.bias_ih_l1.data

    # return module_coll

# =====================================================
