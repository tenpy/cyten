"""Implements a BlockBackend using PyTorch."""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
from numpy import prod
import numpy

from .abelian import AbelianBackend
from .abstract_backend import BlockBackend, Block
from .no_symmetry import NoSymmetryBackend
from .fusion_tree_backend import FusionTreeBackend
from ..dtypes import Dtype

__all__ = ['TorchBlockBackend', 'NoSymmetryTorchBackend', 'AbelianTorchBackend',
           'FusionTreeTorchBackend']


class TorchBlockBackend(BlockBackend):
    """A block-backend using PyTorch"""

    svd_algorithms = ['gesvdj', 'gesvda', 'gesvd']

    def __init__(self, default_device: str = 'cpu') -> None:
        global torch_module
        try:
            import torch
        except ImportError as e:
            raise ImportError('Could not import torch. Use a different backend or install torch.') from e
        torch_module = torch
        self.cyten_dtype_map = {
            torch.float32: Dtype.float32,
            torch.float64: Dtype.float64,
            torch.complex64: Dtype.complex64,
            torch.complex128: Dtype.complex128,
            torch.bool: Dtype.bool,
            None: None,
        }
        self.backend_dtype_map = {
            Dtype.float32: torch.float32,
            Dtype.float64: torch.float64,
            Dtype.complex64: torch.complex64,
            Dtype.complex128: torch.complex128,
            Dtype.bool: torch.bool,
            None: None,
        }
        self.BlockCls = torch.Tensor
        super().__init__(default_device=default_device)
    
    def as_block(self, a, dtype: Dtype = None, return_dtype: bool = False, device: str = None
                 ) -> Block:
        # TODO good error handling if a device does not support a given dtype
        block = torch_module.as_tensor(a, dtype=self.backend_dtype_map[dtype],
                                       device=self.as_device(device))
        if dtype != Dtype.bool:
            block = 1. * block  # force int to float.
        if return_dtype:
            return block, self.cyten_dtype_map[block.dtype]
        return block

    def as_device(self, device: str | None) -> str:
        if device is None:
            device = self.default_device
        res = torch_module.device(device)
        if res.index is None:
            res = torch_module.device(device, index=0)
        return str(res)

    def block_abs_argmax(self, block: Block) -> list[int]:
        flat_idx = torch_module.argmax(torch_module.abs(block))
        # OPTIMIZE numpy has np.unravel_indices. no analogue here?
        idcs = []
        for dim in reversed(block.shape):
            flat_idx, idx = divmod(flat_idx, dim)
            idcs.append(idx)
        return idcs

    def block_add_axis(self, a: Block, pos: int) -> Block:
        return torch_module.unsqueeze(a, pos)

    def block_all(self, a) -> bool:
        return torch_module.all(a)
        
    def block_allclose(self, a: Block, b: Block, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        a = torch_module.as_tensor(a)
        b = torch_module.as_tensor(b)
        a, b = self.to_same_dtype(a, b)
        return torch_module.allclose(a, b, rtol=rtol, atol=atol)

    def block_angle(self, a: Block) -> Block:
        return torch_module.angle(a)

    def block_any(self, a) -> bool:
        return torch_module.any(a)

    def _block_argsort(self, block: Block, axis: int) -> Block:
        return torch_module.argsort(block, dim=axis)

    def block_conj(self, a: Block) -> Block:
        return torch_module.conj(a)

    def block_copy(self, a: Block, device: str = None) -> Block:
        res = a.clone().detach()
        if device is not None:
            res = res.to(self.as_device(device))
        return res

    def block_dtype(self, a: Block) -> Dtype:
        return self.cyten_dtype_map[a.dtype]

    def block_eigh(self, block: Block, sort: str = None) -> tuple[Block, Block]:
        w, v = torch_module.linalg.eigh(block)
        if sort is not None:
            perm = self.block_argsort(w, sort)
            w = w[perm]
            v = v[:, perm]
        return w, v

    def block_eigvalsh(self, block: Block, sort: str = None) -> Block:
        w = torch_module.linalg.eigvalsh(block)
        if sort is not None:
            perm = self.block_argsort(w, sort)
            w = w[perm]
        return w
    
    def block_enlarge_leg(self, block: Block, mask: Block, axis: int) -> Block:
        shape = list(block.shape)
        shape[axis] = len(mask)
        res = torch_module.zeros(shape, dtype=block.dtype, device=block.device)
        idcs = [slice(None, None, None)] * len(shape)
        idcs[axis] = mask
        res[idcs] = block
        return res
    
    def block_exp(self, a: Block) -> Block:
        return torch_module.exp(a)

    def block_from_diagonal(self, diag: Block) -> Block:
        return torch_module.diag(diag)

    def block_from_mask(self, mask: Block, dtype: Dtype) -> Block:
        M, = mask.shape
        N = torch_module.sum(mask)
        res = torch_module.zeros((N, M), dtype=self.backend_dtype_map[dtype])
        res[torch_module.arange(N), mask] = 1
        return res

    def block_from_numpy(self, a: numpy.ndarray, dtype: Dtype = None, device: str = None) -> Block:
        return torch_module.tensor(a, device=self.as_device(device),
                                   dtype=self.backend_dtype_map[dtype])

    def block_get_device(self, a: Block) -> str:
        res = a.device
        if res.index is None:
            res = torch_module.device(res.type, index=0)
        return str(res)

    def block_get_diagonal(self, a: Block, check_offdiagonal: bool) -> Block:
        res = torch_module.diagonal(a)
        if check_offdiagonal:
            if not torch_module.allclose(res, torch_module.diag(a)):
                raise ValueError('Not a diagonal block.')
        return res

    def block_imag(self, a: Block) -> Block:
        if not a.dtype.is_complex:
            return torch_module.zeros_like(a)
        return torch_module.imag(a)

    def block_inner(self, a: Block, b: Block, do_dagger: bool) -> float | complex:
        a, b = self.to_same_dtype(a, b, at_least=torch_module.float16)
        if do_dagger:
            res = torch_module.tensordot(torch_module.conj(a), b, a.ndim)
        else:
            res = torch_module.tensordot(a, b, [tuple(range(a.ndim)), tuple(reversed(range(a.ndim)))])
        return self.block_item(res)

    def block_item(self, a: Block) -> float | complex:
        if a.dtype.is_complex:
            return complex(a)
        else:
            return float(a)

    def block_kron(self, a: Block, b: Block) -> Block:
        a, b = self.to_same_dtype(a, b)
        return torch_module.kron(a, b)

    def block_log(self, a: Block) -> Block:
        return torch_module.log(a)

    def block_max(self, a: Block) -> float | complex:
        return self.block_item(torch_module.max(a))
    
    def block_max_abs(self, a: Block) -> float:
        return self.block_item(torch_module.max(torch_module.abs(a)))

    def block_min(self, a: Block) -> float | complex:
        return self.block_item(torch_module.min(a))

    def block_norm(self, a: Block, order: int | float = 2, axis: int | None = None) -> float:
        res = torch_module.linalg.vector_norm(a, ord=order, dim=axis)
        if axis is None:
            res = self.block_item(res)
        return res
    
    def block_outer(self, a: Block, b: Block) -> Block:
        a, b = self.to_same_dtype(a, b, at_least=torch_module.float16)
        return torch_module.tensordot(a, b, ([], []))

    def block_permute_axes(self, a: Block, permutation: list[int]) -> Block:
        return torch_module.permute(a, permutation)  # TODO: this is documented as a view. is that a problem?

    def block_random_uniform(self, dims: list[int], dtype: Dtype, device: str = None) -> Block:
        return torch_module.rand(*dims, dtype=self.backend_dtype_map[dtype],
                                 device=self.as_device(device))

    def block_random_normal(self, dims: list[int], dtype: Dtype, sigma: float, device: str = None
                            ) -> Block:
        # TODO Note that if device is CUDA, this function synchronizes the device with the CPU
        mean = torch_module.zeros(size=dims, dtype=self.backend_dtype_map[dtype],
                                  device=self.as_device(device))
        std = sigma * torch_module.ones_like(mean, device=device)
        return torch_module.normal(mean, std)

    def block_real(self, a: Block) -> Block:
        return torch_module.real(a)

    def block_real_if_close(self, a: Block, tol: float) -> Block:
        eps = 2.2204460492503131e-16  # TODO make it actually depend on the dtype!!
        if torch_module.all(torch_module.abs(self.block_imag(a)) < tol * eps):
            a = torch_module.real(a)
        return a

    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        torch_module.set_printoptions(linewidth=max_width - len(indent))
        lines = [f'{indent}{line}' for line in repr(a).split('\n')]
        torch_module.set_printoptions(profile='default')
        if len(lines) > max_lines:
            first = (max_lines - 1) // 2
            last = max_lines - 1 - first
            lines = lines[:first] + [f'{indent}...'] + lines[-last:]
        return lines

    def block_reshape(self, a: Block, shape: tuple[int]) -> Block:
        return torch_module.reshape(a, tuple(shape))

    def block_shape(self, a: Block) -> tuple[int]:
        return tuple(a.shape)

    def block_sqrt(self, a: Block) -> Block:
        return torch_module.sqrt(a)

    def block_squeeze_legs(self, a: Block, idcs: list[int]) -> Block:
        # TODO (JU) this is ugly... but torch.squeeze squeezes all axes of dim 1, cant control which
        idx = [0 if ax in idcs else slice(None, None, None) for ax in range(len(a.shape))]
        return a[idx]

    def block_stable_log(self, block: Block, cutoff: float) -> Block:
        return torch_module.where(block > cutoff, torch_module.log(block), 0.)

    def block_sum(self, a: Block, ax: int) -> Block:
        return torch_module.sum(a, ax)

    def block_sum_all(self, a: Block) -> float | complex:
        return self.block_item(torch_module.sum(a))

    def block_tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block:
        a, b = self.to_same_dtype(a, b, at_least=torch_module.float16)
        return torch_module.tensordot(a, b, (idcs_a, idcs_b))

    def block_to_dtype(self, a: Block, dtype: Dtype) -> Block:
        return a.type(self.backend_dtype_map[dtype])

    def block_trace_full(self, a: Block) -> float | complex:
        num_trace = a.ndim // 2
        trace_dim = prod(a.shape[:num_trace])
        perm = [*range(num_trace), *reversed(range(num_trace, 2 * num_trace))]
        a = torch_module.reshape(torch_module.permute(a, perm), (trace_dim, trace_dim))
        return self.block_item(a.diagonal(offset=0, dim1=0, dim2=1).sum(0))

    def block_trace_partial(self, a: Block, idcs1: list[int], idcs2: list[int], remaining: list[int]) -> Block:
        a = torch_module.permute(a, remaining + idcs1 + idcs2)
        trace_dim = int(prod(a.shape[len(remaining):len(remaining)+len(idcs1)]))
        a = torch_module.reshape(a, a.shape[:len(remaining)] + (trace_dim, trace_dim))
        return a.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    def eye_matrix(self, dim: int, dtype: Dtype, device: str = None) -> Block:
        return torch_module.eye(dim, dtype=self.backend_dtype_map[dtype],
                                device=self.as_device(device))

    def get_block_element(self, a: Block, idcs: list[int]) -> complex | float | bool:
        return self.block_item(a[tuple(idcs)])

    def matrix_dot(self, a: Block, b: Block) -> Block:
        a, b = self.to_same_dtype(a, b)
        return torch_module.matmul(a, b)

    def matrix_exp(self, matrix: Block) -> Block:
        raise NotImplementedError  # TODO: could not find a torch implementation via their docs...?

    def matrix_log(self, matrix: Block) -> Block:
        raise NotImplementedError  # TODO: could not find a torch implementation via their docs...?

    def matrix_qr(self, a: Block, full: bool) -> tuple[Block, Block]:
        return torch_module.linalg.qr(a, mode='complete' if full else 'reduced')

    def matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        if a.device.type == 'cuda':
            if algorithm is None:
                algorithm = 'gesvd'
            assert algorithm in self.svd_algorithms
        else:
            if algorithm == 'gesvd':
                algorithm = None
            if algorithm is not None:
                msg = 'For torch, the algorithm keyword is only supported on CUDA hardware'
                raise ValueError(msg)
        U, S, V = torch_module.linalg.svd(a, full_matrices=False, driver=algorithm)
        return U, S, V

    def ones_block(self, shape: list[int], dtype: Dtype, device: str = None) -> Block:
        return torch_module.ones(list(shape), dtype=self.backend_dtype_map[dtype],
                                 device=self.as_device(device))

    def to_same_dtype(self, a: Block, b: Block, at_least=None) -> tuple[Block, ...]:
        # OPTIMIZE is there something built in to torch?
        dtype = torch_module.promote_types(a.dtype, b.dtype)
        if at_least is not None:
            dtype = torch_module.promote_types(dtype, at_least)
        if a.dtype != dtype:
            a = torch_module.as_tensor(a, dtype=dtype)
        if b.dtype != dtype:
            b = torch_module.as_tensor(b, dtype=dtype)
        return a, b

    def synchronize(self):
        """Wait for asynchronous processes (if any) to finish"""
        raise NotImplementedError  # TODO unclear which device to synchronize

    def zero_block(self, shape: list[int], dtype: Dtype, device: str = None) -> Block:
        return torch_module.zeros(list(shape), dtype=self.backend_dtype_map[dtype],
                                  device=self.as_device(device))


class NoSymmetryTorchBackend(NoSymmetryBackend):
    """TODO"""
    
    def __init__(self, default_device: str = 'cpu'):
        block_backend = TorchBlockBackend(default_device=default_device)
        NoSymmetryBackend.__init__(self, block_backend=block_backend)


class AbelianTorchBackend(AbelianBackend):
    """TODO"""
    
    def __init__(self, default_device: str = 'cpu'):
        block_backend = TorchBlockBackend(default_device=default_device)
        AbelianBackend.__init__(self, block_backend=block_backend)


class FusionTreeTorchBackend(FusionTreeBackend):
    """TODO"""
    
    def __init__(self, default_device: str = 'cpu'):
        block_backend = TorchBlockBackend(default_device=default_device)
        FusionTreeBackend.__init__(self, block_backend=block_backend)
