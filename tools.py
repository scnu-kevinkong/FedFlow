# -*- coding: utf-8 -*-
import copy
import torch
import types
import math
import numpy as np
# from scipy import stats # scipy.stats seems unused
import torch.nn.functional as F
import cvxpy as cvx # Ensure cvxpy is installed: pip install cvxpy

def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
    data Indexable (including ability to query length) containing the elements
    Returns:
    Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n):
        # Original code includes (i, i) pairs. FedPAC likely needs distinct pairs j > i ?
        # Let's assume original pairwise (j >= i) was intended.
        for j in range(i, n): # Includes (i, i)
            yield (data[i], data[j])
    # If only distinct pairs (j > i) are needed:
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         yield (data[i], data[j])


def get_protos(protos_dict):
    """
    Calculates the mean prototype for each class from a dictionary of lists of protos.
    Args:
        protos_dict: Dict {class_label: [list of proto tensors]}
    Returns:
        Dict {class_label: mean_proto_tensor}
    """
    protos_mean = {}
    for label, proto_list in protos_dict.items():
        if proto_list: # Check if list is not empty
            # Stack tensors and calculate mean along dim 0
            protos_mean[label] = torch.stack(proto_list).mean(dim=0)
        # else: Handle empty list case? Maybe skip the label or return None/zero tensor?
        # Skipping for now.
    return protos_mean


def protos_aggregation(local_protos_list, local_sizes_list):
    """
    Aggregates local prototypes using weighted averaging based on class sizes.
    Args:
        local_protos_list: List of dictionaries [ {label: proto_tensor}, ... ]
        local_sizes_list: List of tensors [ tensor_counts_per_class, ... ]
    Returns:
        Dictionary {label: aggregated_proto_tensor}
    """
    agg_protos_label = {} # Stores {label: list_of_protos_from_clients}
    agg_sizes_total = {}  # Stores {label: list_of_sizes_for_this_label}

    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx] # Dict for client idx
        local_sizes = local_sizes_list[idx]   # Tensor for client idx

        for label, proto in local_protos.items():
            # Ensure label index is within bounds of local_sizes tensor
            if label < len(local_sizes):
                size = local_sizes[label].item() # Get the count for this specific class
                if size > 0: # Only aggregate if samples exist
                    if label not in agg_protos_label:
                        agg_protos_label[label] = []
                        agg_sizes_total[label] = []
                    agg_protos_label[label].append(proto)
                    agg_sizes_total[label].append(size)
            # else: print(f"Warning: Label {label} out of bounds for client {idx} sizes tensor.")


    # Calculate weighted average for each label
    aggregated_protos = {}
    for label, proto_list in agg_protos_label.items():
        sizes_list = agg_sizes_total[label]
        total_size = sum(sizes_list)

        if total_size > 0 and proto_list:
            # Ensure all tensors are on the same device (e.g., CPU or specified device)
            # Assuming protos in proto_list are already on the target device
            device = proto_list[0].device
            weighted_proto_sum = torch.zeros_like(proto_list[0], device=device)

            for i in range(len(proto_list)):
                 # Ensure size is compatible for broadcasting if needed, though direct scalar multiplication is fine
                weighted_proto_sum += sizes_list[i] * proto_list[i].to(device)

            aggregated_protos[label] = weighted_proto_sum / total_size
        # else: Handle cases with no samples? Return zero proto?

    return aggregated_protos


def average_weights_weighted(local_weights, client_weights, exclude_keys=None):
    """
    Performs weighted averaging of model state dictionaries.
    Args:
        local_weights: A list of state_dicts.
        client_weights: A tensor of weights corresponding to each state_dict in local_weights.
                        Should sum to 1.0 or will be normalized.
        exclude_keys: A list of keys (str) to exclude from averaging.
    Returns:
        A single state_dict with weighted averaged parameters.
    """
    if not local_weights:
        return {} # Return empty dict if no weights to average

    # Normalize client_weights if they don't sum to 1
    total_weight = client_weights.sum()
    if total_weight <= 0:
         # Handle zero weight case - maybe return the first model or raise error
         print("Warning: Total client weight is zero in average_weights_weighted. Returning first model.")
         return copy.deepcopy(local_weights[0])

    normalized_weights = client_weights / total_weight

    w_avg = copy.deepcopy(local_weights[0]) # Initialize with the structure of the first model

    for key in w_avg.keys():
        if exclude_keys and key in exclude_keys:
            # Keep the initial value from w_avg (which is from local_weights[0])
            # Or perhaps better to set it to None or raise error if exclusion means it shouldn't exist?
            # For FedPAC, we typically want the server's *current* classifier weights,
            # but this function averages *client* weights. So exclusion means we don't average this key.
            # Let's keep the value from the first client for structure, assuming server overwrites later.
            continue # Skip averaging for excluded keys

        # Ensure tensors are on the same device for accumulation
        target_device = w_avg[key].device
        # Accumulate weighted average
        w_avg[key].zero_() # Zero out the initial value
        for i in range(len(local_weights)):
            # Check if the key exists in the current client's weights
            if key in local_weights[i]:
                 w_avg[key] += normalized_weights[i].item() * local_weights[i][key].to(target_device)
            # else: Handle missing key? Maybe skip this client for this key?

    return w_avg


def agg_classifier_weighted_p(local_weights, client_weights_alpha, classifier_keys, target_client_idx_in_active):
    """
    Aggregates only the classifier weights using personalized weights alpha_i.
    Args:
        local_weights: List of state_dicts from all *participating* clients in the round.
        client_weights_alpha: Numpy array or list of personalized weights (alpha_i) for the target client.
                              Should sum to 1.0. Length must match len(local_weights).
        classifier_keys: List of keys belonging to the classifier layers.
        target_client_idx_in_active: The index of the client for whom this aggregation is being done,
                                     within the context of the local_weights list. This is needed
                                     if the structure needs to be copied from the target client initially.
                                     (Although we are overwriting the keys being aggregated).
    Returns:
        A state_dict containing only the aggregated classifier parameters.
    """
    if not local_weights or client_weights_alpha is None or not classifier_keys:
        # Return the target client's own weights if inputs are invalid or aggregation fails
        print("Warning: Invalid input to agg_classifier_weighted_p. Returning target client's weights.")
        # Check if target_client_idx_in_active is valid
        if 0 <= target_client_idx_in_active < len(local_weights):
             return {k: v for k, v in local_weights[target_client_idx_in_active].items() if k in classifier_keys}
        else:
             return {} # Or raise error

    num_participants = len(local_weights)
    if len(client_weights_alpha) != num_participants:
         raise ValueError("Length of client_weights_alpha must match the number of local_weights.")

    # Initialize structure using the target client's state dict, but only for classifier keys
    w_agg = {key: copy.deepcopy(local_weights[target_client_idx_in_active][key]) for key in classifier_keys if key in local_weights[target_client_idx_in_active]}

    # Check if client_weights_alpha is numpy array, convert if list
    if isinstance(client_weights_alpha, list):
         alpha = np.array(client_weights_alpha)
    else:
         alpha = client_weights_alpha

    # Normalize alpha if it doesn't sum to 1 (it should ideally)
    alpha_sum = alpha.sum()
    if alpha_sum <= 0:
        print("Warning: Personalized weights alpha sum to zero or less. Aggregation may be invalid.")
        # Fallback: return target client's own weights
        return {k: v for k, v in local_weights[target_client_idx_in_active].items() if k in classifier_keys}

    if not np.isclose(alpha_sum, 1.0):
        print(f"Warning: Personalized weights alpha sum to {alpha_sum}, normalizing.")
        alpha = alpha / alpha_sum

    # Perform weighted aggregation for classifier keys
    for key in classifier_keys:
         if key in w_agg: # Ensure key exists in the target client's structure
             target_device = w_agg[key].device
             w_agg[key].zero_() # Zero out before accumulation
             for i in range(num_participants):
                 if alpha[i] > 0 and key in local_weights[i]: # Only add if weight > 0 and key exists
                     w_agg[key] += alpha[i] * local_weights[i][key].to(target_device)

    return w_agg


def get_head_agg_weight(num_users, Vars, Hs, device='cpu'):
    """
    Calculates personalized classifier aggregation weights (alpha) using QP.
    Args:
        num_users: Number of participating users (m).
        Vars: List of scalar variances [V_1, V_2, ..., V_m].
        Hs: List of tensors [H_1, H_2, ..., H_m], where each H_k is [num_classes, d].
        device: The torch device to perform calculations on.
    Returns:
        List of numpy arrays [alpha_1, alpha_2, ..., alpha_m], where each alpha_i is the
        set of weights for aggregating classifiers for client i. Returns None for a client
        if QP fails.
    """
    if not Vars or not Hs or len(Vars) != num_users or len(Hs) != num_users:
        print("Error: Invalid input Vars or Hs for get_head_agg_weight.")
        return [None] * num_users # Return None for all users if input is bad

    # Ensure Hs tensors are on the specified device
    Hs_tensor = [h.to(device) for h in Hs]
    # Ensure Vars is a tensor on the specified device
    v_tensor = torch.tensor(Vars, device=device) # Variance term V

    num_cls = Hs_tensor[0].shape[0] # number of classes
    d = Hs_tensor[0].shape[1] # dimension of feature representation

    all_alphas = []

    for i in range(num_users): # Calculate weights alpha_i for target client i
        h_ref = Hs_tensor[i] # H_i for the target client

        # Calculate pairwise distance matrix based on H vectors
        dist = torch.zeros((num_users, num_users), device=device)
        # Using range(num_users) for indices j1, j2 corresponding to Vars/Hs lists
        for j1 in range(num_users):
             for j2 in range(j1, num_users): # Includes j1==j2
                 h_j1 = Hs_tensor[j1]
                 h_j2 = Hs_tensor[j2]
                 # Calculate sum_k <H_i^k - H_j1^k, H_i^k - H_j2^k>
                 # term = torch.sum((h_ref - h_j1) * (h_ref - h_j2)) # Element-wise product and sum
                 # Or using the original code's explicit loop:
                 h = torch.zeros((d, d), device=device) # Temporary matrix for trace calculation
                 for k in range(num_cls):
                     # Reshape ensures correct outer product for trace calculation
                     diff1 = (h_ref[k] - h_j1[k]).reshape(d, 1)
                     diff2 = (h_ref[k] - h_j2[k]).reshape(1, d)
                     h += torch.mm(diff1, diff2)
                 dj12 = torch.trace(h)

                 dist[j1, j2] = dj12
                 dist[j2, j1] = dj12 # Symmetric matrix

        # Construct the P matrix for the QP problem: P = diag(V) + dist
        p_matrix = torch.diag(v_tensor) + dist
        p_matrix_np = p_matrix.cpu().numpy()  # Convert to numpy for CVXPY

        # --- Try solving the QP problem ---
        alpha = None # Default to None if solver fails
        try:
            alphav = cvx.Variable(num_users) # Variable vector alpha_i
            # Objective: Minimize alpha_i^T @ P @ alpha_i
            # REMOVED psd_wrap: Pass the matrix directly. CVXPY might warn/error if not PSD.
            objective = cvx.Minimize(cvx.quad_form(alphav, p_matrix_np))
            # Constraints: sum(alpha_i) = 1, alpha_i >= 0
            constraints = [cvx.sum(alphav) == 1.0, alphav >= 0]
            prob = cvx.Problem(objective, constraints)

            # Try solving with a solver known to handle non-PSD (e.g., SCS, OSQP - check CVXPY docs)
            # prob.solve(solver=cvx.SCS)
            prob.solve() # Use default solver first

            if prob.status in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
                alpha = alphav.value
                # Clean up small values and ensure non-negativity/summation
                eps = 1e-6 # Epsilon for numerical stability
                alpha = np.maximum(alpha, 0)       # Ensure non-negativity
                alpha_sum = np.sum(alpha)
                if alpha_sum > eps : # Avoid division by zero if sum is too small
                     alpha = alpha / alpha_sum    # Normalize
                else:
                     print(f"Warning: Sum of alpha for client {i} is near zero ({alpha_sum}). Resetting to local.")
                     alpha = np.zeros(num_users)
                     alpha[i] = 1.0 # Fallback to local weight

                alpha[alpha < eps] = 0             # Zero-out tiny values
                # Re-normalize again after zeroing out small values
                alpha_sum_final = np.sum(alpha)
                if alpha_sum_final > eps:
                    alpha = alpha / alpha_sum_final
                else:
                     print(f"Warning: Final sum of alpha for client {i} is near zero ({alpha_sum_final}). Resetting to local.")
                     alpha = np.zeros(num_users)
                     alpha[i] = 1.0 # Fallback to local weight


                # Optional: Print weights for debugging
                # if i == 0:
                #     print(f'({i+1}) Agg Weights of Classifier Head (No psd_wrap)')
                #     print(alpha, '\n')

            else:
                print(f"CVXPY solver failed for user {i} with status: {prob.status}. Matrix P might not be PSD or other solver issue.")
                # Fallback to local weights if optimization fails
                alpha = np.zeros(num_users)
                alpha[i] = 1.0

        except cvx.SolverError as e:
            print(f"CVXPY SolverError for user {i}: {e}. Using local weights.")
            alpha = np.zeros(num_users)
            alpha[i] = 1.0
        except Exception as e:
            print(f"Error during CVXPY solve for user {i}: {e}. Using local weights.")
            alpha = np.zeros(num_users)
            alpha[i] = 1.0

        all_alphas.append(alpha) # Append the resulting numpy array (or None if error occurred before fallback)

    return all_alphas


# --------------------------------------------------------------------- #
# Gradient access utility functions (seem unused by FedPAC logic above)
# Keep them if they are used elsewhere in your project.
# --------------------------------------------------------------------- #

def grad_of(tensor):
    """ Get the gradient of a given tensor, make it zero if missing. """
    grad = tensor.grad
    if grad is not None:
        return grad
    grad = torch.zeros_like(tensor)
    # tensor.grad = grad # Avoid modifying tensor state if just reading
    return grad

def grads_of(tensors):
    """ Iterate of the gradients of the given tensors. """
    return (grad_of(tensor) for tensor in tensors)

# ---------------------------------------------------------------------------- #
# "Flatten" and "relink" operations (seem unused by FedPAC logic above)
# Keep them if they are used elsewhere in your project.
# ---------------------------------------------------------------------------- #

def relink(tensors, common):
    """ Relink tensors to a contiguous memory segment. """
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    pos = 0
    for tensor in tensors:
        npos = pos + tensor.numel()
        # This modifies the original tensors' data pointers. Use with caution.
        try: # Ensure view is compatible
             tensor.data = common[pos:npos].view(*tensor.shape)
        except RuntimeError as e:
             print(f"Error relinking tensor with shape {tensor.shape} at pos {pos}:{npos}. Common size: {common.numel()}. Error: {e}")
             # Handle error appropriately
             raise e
        pos = npos
    # common.linked_tensors = tensors # Adding attributes dynamically can be risky
    return common

def flatten(tensors):
    """ Flatten tensors into a single contiguous tensor. """
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    if not tensors:
         return torch.tensor([]) # Handle empty input

    # Flatten each tensor and concatenate
    flat_tensors = [tensor.detach().view(-1) for tensor in tensors] # Detach to avoid grad issues?
    common = torch.cat(flat_tensors)
    # return relink(tensors, common) # Relinking modifies original tensors, might not be desired.
    # Returning just the flat tensor is usually safer.
    return common

# ---------------------------------------------------------------------------- #
# Functions to get/set gradients/parameters as flat vectors (seem unused by FedPAC logic)
# Keep if used elsewhere.
# ---------------------------------------------------------------------------- #

def get_gradient_values(model):
    """ Get all gradients concatenated into a single flat vector. """
    if not list(model.parameters()): return torch.tensor([])
    grads = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
    if not grads: return torch.tensor([])
    return torch.cat(grads)

def set_gradient_values(model, gradient):
    """ Set model gradients from a flat vector. """
    if not list(model.parameters()): return
    cur_pos = 0
    for param in model.parameters():
        numel = param.numel()
        if param.requires_grad: # Only set gradients for params that require them
            if param.grad is None:
                 param.grad = torch.zeros_like(param.data) # Initialize grad if None
            # Get the segment from the flat gradient
            grad_segment = gradient[cur_pos : cur_pos + numel].view(param.size())
            param.grad.copy_(grad_segment) # Copy values into existing grad tensor
        cur_pos += numel
        if cur_pos > len(gradient): # Check bounds
             print("Warning: Gradient vector shorter than required for model parameters.")
             break


def get_parameter_values(model):
    """ Get all parameters concatenated into a single flat vector. """
    if not list(model.parameters()): return torch.tensor([])
    params = [p.data.detach().view(-1) for p in model.parameters()]
    return torch.cat(params)

def set_parameter_values(model, parameter):
    """ Set model parameters from a flat vector. """
    if not list(model.parameters()): return
    cur_pos = 0
    for param in model.parameters():
        numel = param.numel()
        # Get the segment from the flat parameter vector
        param_segment = parameter[cur_pos : cur_pos + numel].view(param.size())
        param.data.copy_(param_segment) # Copy values into existing data tensor
        cur_pos += numel
        if cur_pos > len(parameter): # Check bounds
             print("Warning: Parameter vector shorter than required for model parameters.")
             break
# ---------------------------------------------------------------------------- #