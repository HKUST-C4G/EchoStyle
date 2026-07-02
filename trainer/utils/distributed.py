import functools
import logging
import pickle

import numpy as np
import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


#################### Basic Operations ####################


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size(group=None):
    return dist.get_world_size(group) if is_dist_initialized() else 1


def get_rank(group=None):
    return dist.get_rank(group) if is_dist_initialized() else 0


# TODO: Simplify this function.
def all_gather(tensor, uniform_size=True, group=None, **kwargs):
    world_size = get_world_size(group)
    if world_size == 1:
        return [tensor]
    assert tensor.is_contiguous(), "ops.all_gather requires the tensor to be contiguous()"

    if uniform_size:
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group, **kwargs)
        return tensor_list
    else:
        # collect tensor shapes across GPUs
        shape = tuple(tensor.shape)
        shape_list = generalized_all_gather(shape, group)

        # flatten the tensor
        tensor = tensor.reshape(-1)
        size = int(np.prod(shape))
        size_list = [int(np.prod(u)) for u in shape_list]
        max_size = max(size_list)

        # pad to maximum size
        if size != max_size:
            padding = tensor.new_zeros(max_size - size)
            tensor = torch.cat([tensor, padding], dim=0)

        # all_gather
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group, **kwargs)

        # reshape tensors
        tensor_list = [t[:n].view(s) for t, n, s in zip(tensor_list, size_list, shape_list)]
        return tensor_list


@functools.lru_cache()
def get_global_gloo_group():
    backend = dist.get_backend()
    assert backend in ["gloo", "nccl"]
    if backend == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device{}".format(
                get_rank(), len(buffer) / (1024**3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, "gather/all_gather must be called from ranks withinthe give group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]

    # gather tensors and compute the maximum size
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # pad tensors to the same size
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def generalized_all_gather(data, group=None):
    if get_world_size(group) == 1:
        return [data]
    if group is None:
        group = get_global_gloo_group()

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving tensors from all ranks
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size(group)
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


################ Differentiable Operations ###############


def _split(input, dim, group):
    # skip if world_size == 1
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)
    if world_size == 1:
        return input

    # split sequence
    assert input.size(dim) % world_size == 0, f"input.size(dim) {input.size(dim)}, world_size{world_size}"
    return input.chunk(world_size, dim=dim)[rank].contiguous()


def _gather(input, dim, group):
    # skip if world_size == 1
    world_size = get_world_size(group=group)
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input, uniform_size=True, group=group)
    return torch.cat(output, dim=dim).contiguous()


class AllToAll(Function):
    @staticmethod
    def forward(ctx, input, scatter_dim, gather_dim, group):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        return all_to_all(input, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        return (all_to_all(grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group), None, None, None)


class SplitForwardGatherBackward(Function):
    @staticmethod
    def forward(ctx, input, dim, group=None, grad_scale=None):
        ctx.dim = dim
        ctx.group = group
        ctx.grad_scale = grad_scale
        return _split(input, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * get_world_size(group=ctx.group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / get_world_size(group=ctx.group)
        return _gather(grad_output, ctx.dim, ctx.group), None, None, None


class GatherForwardSplitBackward(Function):
    @staticmethod
    def forward(ctx, input, dim, group=None, grad_scale=None):
        ctx.dim = dim
        ctx.group = group
        ctx.grad_scale = grad_scale
        return _gather(input, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * get_world_size(group=ctx.group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / get_world_size(group=ctx.group)
        return _split(grad_output, ctx.dim, ctx.group), None, None, None


def diff_all_to_all(input, scatter_dim, gather_dim, group=None):
    return AllToAll.apply(input, scatter_dim, gather_dim, group)


def split_forward_gather_backward(input, dim, group=None, grad_scale=None):
    return SplitForwardGatherBackward.apply(input, dim, group, grad_scale)


def gather_forward_split_backward(input, dim, group=None, grad_scale=None):
    return GatherForwardSplitBackward.apply(input, dim, group, grad_scale)


##################### Process Groups #####################

DEVICE_MESH = None


def init_process_groups(world_size, sequence_parallel_size=1):
    """
    Initialize data / sequence parallel groups.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we use 8 GPUs to parallelize sequence.
    `init_device_mesh` will create 8 data parallel groups, and 2 sequence parallel groups as:
        8 data parallel groups:
            [g0, g8], [g1, g9], [g2, g10], [g3, g11],
            [g4, g12], [g5, g13], [g6, g14], [g7, g15]
        2 sequence parallel groups:
            [g0, g1, g2, g3, g4, g5, g6, g7],
            [g8, g9, g10, g11, g12, g13, g14, g15]

    Args:
        sequence_parallel_size: Number of GPUs used to parallelize sequence.
    """
    # Parallel group sizes.
    assert world_size % sequence_parallel_size == 0, (
        f"world_size {world_size} is not divisible by sequence_parallel_size {sequence_parallel_size}"
    )
    data_parallel_size = world_size // sequence_parallel_size
    logger.info(f"Initialize data parallel with size {data_parallel_size}")
    logger.info(f"Initialize sequence parallel with size {sequence_parallel_size}")

    global DEVICE_MESH
    # torch.cuda.set_device(local_rank) will be called by init_device_mesh.
    DEVICE_MESH = init_device_mesh("cuda", (data_parallel_size, sequence_parallel_size), mesh_dim_names=("dp", "sp"))


def get_device_mesh():
    return DEVICE_MESH


def get_sequence_parallel_group():
    return DEVICE_MESH["sp"].get_group()


def get_sequence_parallel_rank():
    return dist.get_rank(group=get_sequence_parallel_group())


def get_sequence_parallel_world_size():
    return dist.get_world_size(group=get_sequence_parallel_group())


def get_data_parallel_group():
    return DEVICE_MESH["dp"].get_group()


def get_data_parallel_rank():
    return dist.get_rank(group=get_data_parallel_group())


def get_data_parallel_world_size():
    return dist.get_world_size(group=get_data_parallel_group())
