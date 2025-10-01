from torch.nn.functional import normalize
from torch import zeros, mm, cat, tensor, where, unique, repeat_interleave, vstack
from scipy.cluster.hierarchy import linkage, fcluster

# ====================================================================

def fibration_linear(weights, in_clusters, threshold, first_layer = False, bias = None):
	dim_out, dim_in  = weights.shape

	if first_layer:
		collapse_weights = weights
	else:
		idx_in_clusters  = unique(in_clusters)
		num_in_clusters  = len(idx_in_clusters)
		collapse_weights = zeros((dim_out, num_in_clusters))

		for pos_color, color in enumerate(idx_in_clusters): 
		    indices_k = where(in_clusters == color)[0]
		    collapse_weights[:, pos_color] = weights[:, indices_k].sum(axis=1)

	if bias is not None:
		dev_ = collapse_weights.device
		collapse_weights = cat((collapse_weights, bias.to(dev_).unsqueeze(1)), dim=1)

	collapse_weights_norm = normalize(collapse_weights, dim=1)
	distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	Z = linkage(distance, method='average')
	clusters = tensor(fcluster(Z, t=threshold, criterion='distance'))

	return clusters

# ====================================================================

def fibration_conv2d(weights, in_clusters, threshold, first_layer = False, bias = None):
	out_n, in_n, hx, hy = weights.shape
	weights = weights.view(out_n, in_n, -1)

	if first_layer:
		collapse_weights = weights.view(out_n, -1)
	else:
		idx_in_clusters  = unique(in_clusters)
		num_in_clusters  = len(idx_in_clusters)
		collapse_weights = zeros((out_n, num_in_clusters, hx*hy))

		for pos_color, color in enumerate(idx_in_clusters): 
		    indices_k = where(in_clusters == color)[0]
		    collapse_weights[:, pos_color, :] = weights[:, indices_k, :].sum(axis=1)

		collapse_weights = collapse_weights.view(out_n,-1)

	if bias is not None:
		dev_ = collapse_weights.device
		collapse_weights = cat((collapse_weights, bias.to(dev_).unsqueeze(1)), dim=1)

	collapse_weights_norm = normalize(collapse_weights, dim=1)
	distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	Z = linkage(distance, method='average')
	clusters = tensor(fcluster(Z, t=threshold, criterion='distance'))

	return clusters

# ====================================================================

def fibration_flatten(in_clusters, times):
	return repeat_interleave(in_clusters,times)

# ====================================================================

def fibration_lstm(weight_ih, weight_hh, in_clusters, threshold, T=5):
	dim_h, dim_in = weight_ih.shape

	h_clusters = zeros(dim_h)
	c_clusters = zeros(dim_h)

	for t in range(T):

		# (1) Collapse W_ii,...W_io based on in_cluster.
		idx_in_clusters  = unique(in_clusters)
		num_in_clusters = len(idx_in_clusters)
		collapse_weights_i = zeros((dim_h, num_in_clusters))

		for pos_color, color in enumerate(idx_in_clusters): 
			indices_k = where(in_clusters == color)[0]
			collapse_weights_i[:, pos_color] = weight_ih[:, indices_k].sum(axis=1)

		# (2) Collapse W_hi,...W_ho based on h_cluster.

		idx_h_clusters  = unique(h_clusters)
		num_h_clusters = len(idx_h_clusters)
		collapse_weights_h = zeros((dim_h, num_h_clusters))

		for pos_color, color in enumerate(idx_h_clusters): 
			indices_k = where(h_clusters == color)[0]
			print(indices_k)
			collapse_weights_h[:, pos_color] = weight_hh[:, indices_k].sum(axis=1)

		collapse_weights = cat((collapse_weights_i,collapse_weights_h),dim=1)

		# (3) Clusters of the gates

		collapse_weights_norm = normalize(collapse_weights, dim=1)
		distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
		distance = distance.cpu().numpy()

		Z = linkage(distance, method='average')
		gates_clusters = tensor(fcluster(Z, t=threshold, criterion='distance'))

		# (4) (F,C,I,G) clusters
		gates_cluster_matrix = gates_clusters.reshape(4,dim_h)

		# (5) c_clusters
		new_c_clusters = vstack((gates_cluster_matrix[:3,:], c_clusters))
		_, c_clusters = unique(new_c_clusters.T, dim=0, return_inverse=True)

		# (6) h_clusters
		new_h_clusters = vstack((gates_cluster_matrix[3,:], c_clusters))
		_, h_clusters = unique(new_h_clusters.T, dim=0, return_inverse=True)
	
	return h_clusters