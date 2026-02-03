import torch
from itertools import product

# branch_num = 2
# depth = 16
# dflash_choice = []
# for d in range(1, depth + 1):
#     dflash_choice.extend(list(product(range(branch_num), repeat=d)))

# dflash_choice = [
#     (0,), (1,), (0, 0), (2,), (0, 1), (3,), (1, 0), (4,), (0, 2), (5,),
#     (0, 3), (0, 0, 0), (6,), (0, 4), (2, 0), (7,), (1, 1), (0, 5), (3, 0), (8,),
#     (9,), (0, 6), (0, 7), (0, 0, 1), (1, 2), (4, 0), (0, 1, 0), (0, 8), (0, 9), (2, 1),
#     (0, 0, 2), (5, 0), (1, 3), (0, 0, 3), (1, 0, 0), (1, 4), (6, 0), (0, 2, 0), (3, 1), (2, 2),
#     (0, 0, 4), (7, 0), (0, 1, 1), (1, 5), (4, 1), (0, 0, 5), (0, 3, 0), (9, 0), (8, 0), (1, 6),
#     (0, 0, 6), (2, 3), (0, 1, 2), (3, 2), (0, 4, 0), (2, 0, 0), (1, 7), (1, 0, 1), (0, 0, 7), (5, 1),
#     (2, 4), (0, 0, 8), (0, 2, 1)
# ]
dflash_choice = [tuple([0] * (i + 1)) for i in range(16)] + [tuple([1] * (i + 1)) for i in range(16)]
TOPK=5

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.
    
    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask,
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        }
    
    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v,  device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers




def generate_candidates(draft_ids, verified_ids, tree_indices, retrieve_indices):

    bs, seq, topk = draft_ids.shape
    # draft_ids: to (bs, [top1-topk], seq)
    candidates = draft_ids.permute(0, 2, 1).contiguous().view(bs, -1)
    candidates = torch.cat([verified_ids, candidates], dim=1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[:, tree_indices]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((bs, 1), dtype=torch.long, device=tree_candidates.device)], dim=1)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[:, retrieve_indices]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates
    return cart_candidates, tree_candidates

if __name__ == "__main__":
    print(dflash_choice)
    buffers = generate_medusa_buffers(dflash_choice, device="cpu")
    print(buffers["medusa_attn_mask"].shape)
    for k, v in buffers.items():
        print(f"{k}: {v}")