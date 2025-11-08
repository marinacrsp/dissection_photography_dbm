"""
Helpers for distributed training.
"""

import os

import torch
import torch.distributed as dist

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
WORLD_RANK = int(os.environ.get("RANK", "0"))

def setup_dist():
    if dist.is_initialized():
        return

    # Check if running in distributed mode
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("[dist_util] Running in non-distributed mode.")
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    print(f"[dist_util] Initializing distributed mode with backend={backend}, local_rank={local_rank}")
    dist.init_process_group(backend=backend)
    
    # Optional: avoid warning by setting correct device
    dist.barrier(device_ids=[local_rank])
def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")