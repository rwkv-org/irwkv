# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def get_load_balance_assignment(
    sizes: list[int],
    num_gpus: int = 2,
) -> tuple[list[int], list[int], list[int]]:
    """Generate load balancing assignment and metadata
    for distributing data across GPUs.
    The load is determined by the total image sizes,
    not the number of images.

    Args:
        sizes: The size of each image
        num_gpus: Number of GPUs to balance across

    Returns:
        shuffle_indices:
            Indices to reorder data for balanced loading
        gpu_sample_counts:
            Number of samples assigned to each GPU
        grouped_sizes_per_gpu:
            Total size assigned to each GPU

    Example:
        ```
        sizes = [1000, 100, 200, 50]
        num_gpus = 2
        ```

    """
    n_samples = len(sizes)

    # Handle edge cases
    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    # Use greedy algorithm - balance by total size, not sample count
    gpu_assignments = [list[int]() for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus  # This tracks total SIZE, not sample count

    # Sort indices by size (largest first for better load balancing)
    # sizes = [1000, 100, 200, 50]
    # large_to_small_indices = [0, 2, 1, 3]
    large_to_small_indices = sorted(
        range(n_samples), key=lambda i: sizes[i], reverse=True
    )

    for idx in large_to_small_indices:
        # Find GPU with minimum current load (by total size)
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    # Create shuffle indices and counts
    shuffle_indices = list[int]()
    gpu_sample_counts = list[int]()
    for gpu_id in range(num_gpus):
        # GPU_0 = [1000] = [0]
        # GPU_1 = [200, 100, 50] = [2, 1, 3]
        # shuffle_indices = [0, 2, 1, 3]
        shuffle_indices.extend(gpu_assignments[gpu_id])
        # GPU_0 = [1]
        # GPU_1 = [3]
        # gpu_sample_counts = [1, 3]
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return (shuffle_indices, gpu_sample_counts, gpu_loads)
