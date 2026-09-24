"""
GPU-Accelerated Optimizers for Slang-KANs training.
Includes SGD, Adam, AdamW, Muon, RMSprop, and Lion with Slang GPU kernels
and seamless NumPy CPU fallback.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Sequence, Tuple, List, Optional, Union, Dict, Any

try:
    import slangpy
except ImportError:
    slangpy = None


def _zeropower_via_newtonschulz5(G: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray:
    """
    Newton-Schulz iteration (order 5) to compute the approximate matrix sign / orthogonalization.
    Used by the Muon optimizer (MomentUm Orthogonalized by Newton-schulz).
    Formula coefficients: a=3.4445, b=-4.7750, c=2.0315
    """
    assert G.ndim == 2, f"Expected 2D matrix for Newton-Schulz, got {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315

    transpose = G.shape[0] > G.shape[1]
    X = G.T if transpose else G.copy()

    # Spectral norm normalization
    norm = np.linalg.norm(X) + eps
    X = X / norm

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transpose:
        X = X.T
    return X


class Optimizer:
    """Base class for Slang-KAN optimizers."""
    def __init__(self, params: Sequence[Tuple[np.ndarray, np.ndarray]], lr: float = 1e-3, use_gpu: bool = True):
        self.params = list(params)
        self.lr = float(lr)
        self.use_gpu = bool(use_gpu)
        self._gpu_buffers: Dict[int, Any] = {}

    def zero_grad(self) -> None:
        for param, grad in self.params:
            if grad is not None:
                grad.fill(0)

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum, Nesterov acceleration, and weight decay.
    Accelerated with Slang compute shaders.
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_gpu: bool = True,
    ):
        super().__init__(params, lr, use_gpu=use_gpu)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.velocities: List[Optional[np.ndarray]] = [
            np.zeros_like(p) if momentum > 0.0 else None for p, _ in self.params
        ]

    def step(self) -> None:
        mgr = None
        if self.use_gpu:
            try:
                from slang_kans.device import get_device
                mgr = get_device()
                if not mgr.is_gpu_available():
                    mgr = None
            except Exception:
                mgr = None

        if mgr is not None:
            dev = mgr.device
            kernel = mgr.get_or_compile_kernel("optim", "sgd_step")
            for i, (param, grad) in enumerate(self.params):
                if grad is None:
                    continue
                if param.dtype == np.float32 and grad.dtype == np.float32:
                    N = param.size
                    p_key = id(param)
                    v = self.velocities[i]
                    if v is None:
                        v = np.zeros_like(param)
                        self.velocities[i] = v

                    if p_key not in self._gpu_buffers:
                        buf_p = dev.create_buffer(data=np.ascontiguousarray(param), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        buf_g = dev.create_buffer(data=np.ascontiguousarray(grad), usage=slangpy.BufferUsage.shader_resource)
                        buf_v = dev.create_buffer(data=np.ascontiguousarray(v), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        self._gpu_buffers[p_key] = (buf_p, buf_g, buf_v)
                    else:
                        buf_p, buf_g, buf_v = self._gpu_buffers[p_key]
                        buf_g.copy_from_numpy(np.ascontiguousarray(grad))

                    kernel.dispatch(
                        thread_count=[N, 1, 1],
                        param=buf_p,
                        grad=buf_g,
                        v=buf_v,
                        lr=self.lr,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay,
                        nesterov=1 if self.nesterov else 0,
                        n_elems=N
                    )
                    dev.wait_for_idle()
                    param[:] = buf_p.to_numpy().view(np.float32)[:N].reshape(param.shape)
                    continue

        # CPU Fallback
        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            g = grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * param

            if self.momentum != 0.0:
                v = self.velocities[i]
                if v is None:
                    v = np.zeros_like(param)
                    self.velocities[i] = v
                v[:] = self.momentum * v + g
                if self.nesterov:
                    update = g + self.momentum * v
                else:
                    update = v
            else:
                update = g

            param -= self.lr * update


class Adam(Optimizer):
    """
    Standard Adam optimizer with L2 weight decay (coupled).
    Accelerated with Slang compute shaders.
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_gpu: bool = True,
    ):
        super().__init__(params, lr, use_gpu=use_gpu)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0

        self.m = [np.zeros_like(p) for p, _ in self.params]
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def step(self) -> None:
        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t
        lr_t = self.lr * math.sqrt(bias_correction2) / bias_correction1

        mgr = None
        if self.use_gpu:
            try:
                from slang_kans.device import get_device
                mgr = get_device()
                if not mgr.is_gpu_available():
                    mgr = None
            except Exception:
                mgr = None

        if mgr is not None:
            dev = mgr.device
            kernel = mgr.get_or_compile_kernel("optim", "adam_step")
            for i, (param, grad) in enumerate(self.params):
                if grad is None:
                    continue
                if param.dtype == np.float32 and grad.dtype == np.float32:
                    N = param.size
                    p_key = id(param)
                    if p_key not in self._gpu_buffers:
                        buf_p = dev.create_buffer(data=np.ascontiguousarray(param), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        buf_g = dev.create_buffer(data=np.ascontiguousarray(grad), usage=slangpy.BufferUsage.shader_resource)
                        buf_m = dev.create_buffer(data=np.ascontiguousarray(self.m[i]), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        buf_v = dev.create_buffer(data=np.ascontiguousarray(self.v[i]), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        self._gpu_buffers[p_key] = (buf_p, buf_g, buf_m, buf_v)
                    else:
                        buf_p, buf_g, buf_m, buf_v = self._gpu_buffers[p_key]
                        buf_g.copy_from_numpy(np.ascontiguousarray(grad))

                    kernel.dispatch(
                        thread_count=[N, 1, 1],
                        param=buf_p,
                        grad=buf_g,
                        m=buf_m,
                        v=buf_v,
                        lr=self.lr,
                        beta1=self.beta1,
                        beta2=self.beta2,
                        eps=self.eps,
                        weight_decay=self.weight_decay,
                        lr_t=lr_t,
                        n_elems=N
                    )
                    dev.wait_for_idle()
                    param[:] = buf_p.to_numpy().view(np.float32)[:N].reshape(param.shape)
                    continue

        # CPU Fallback
        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            g = grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * param

            m = self.m[i]
            v = self.v[i]

            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            denom = np.sqrt(v) + self.eps
            param -= lr_t * (m / denom)


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, 2019).
    Accelerated with Slang compute shaders.
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        use_gpu: bool = True,
    ):
        super().__init__(params, lr, use_gpu=use_gpu)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0

        self.m = [np.zeros_like(p) for p, _ in self.params]
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def step(self) -> None:
        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t
        lr_t = self.lr * math.sqrt(bias_correction2) / bias_correction1

        mgr = None
        if self.use_gpu:
            try:
                from slang_kans.device import get_device
                mgr = get_device()
                if not mgr.is_gpu_available():
                    mgr = None
            except Exception:
                mgr = None

        if mgr is not None:
            dev = mgr.device
            kernel = mgr.get_or_compile_kernel("optim", "adamw_step")
            for i, (param, grad) in enumerate(self.params):
                if grad is None:
                    continue
                if param.dtype == np.float32 and grad.dtype == np.float32:
                    N = param.size
                    p_key = id(param)
                    if p_key not in self._gpu_buffers:
                        buf_p = dev.create_buffer(data=np.ascontiguousarray(param), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        buf_g = dev.create_buffer(data=np.ascontiguousarray(grad), usage=slangpy.BufferUsage.shader_resource)
                        buf_m = dev.create_buffer(data=np.ascontiguousarray(self.m[i]), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        buf_v = dev.create_buffer(data=np.ascontiguousarray(self.v[i]), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        self._gpu_buffers[p_key] = (buf_p, buf_g, buf_m, buf_v)
                    else:
                        buf_p, buf_g, buf_m, buf_v = self._gpu_buffers[p_key]
                        buf_g.copy_from_numpy(np.ascontiguousarray(grad))

                    kernel.dispatch(
                        thread_count=[N, 1, 1],
                        param=buf_p,
                        grad=buf_g,
                        m=buf_m,
                        v=buf_v,
                        lr=self.lr,
                        beta1=self.beta1,
                        beta2=self.beta2,
                        eps=self.eps,
                        weight_decay=self.weight_decay,
                        lr_t=lr_t,
                        n_elems=N
                    )
                    dev.wait_for_idle()
                    param[:] = buf_p.to_numpy().view(np.float32)[:N].reshape(param.shape)
                    continue

        # CPU Fallback
        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            if self.weight_decay != 0.0:
                param -= self.lr * self.weight_decay * param

            g = grad
            m = self.m[i]
            v = self.v[i]

            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            denom = np.sqrt(v) + self.eps
            param -= lr_t * (m / denom)


class Muon(Optimizer):
    """
    Muon (MomentUm Orthogonalized by Newton-schulz) optimizer.
    Uses 5th-order Newton-Schulz polynomial iteration to orthogonalize updates on 2D weight matrices,
    providing exceptional convergence speed on neural networks and KAN layers.
    Accelerated with Slang compute shaders.
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
        use_gpu: bool = True,
    ):
        super().__init__(params, lr, use_gpu=use_gpu)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.ns_steps = int(ns_steps)
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def step(self) -> None:
        mgr = None
        if self.use_gpu:
            try:
                from slang_kans.device import get_device
                mgr = get_device()
                if not mgr.is_gpu_available():
                    mgr = None
            except Exception:
                mgr = None

        if mgr is not None:
            dev = mgr.device
            k_mom = mgr.get_or_compile_kernel("optim", "muon_momentum_update")
            k_param = mgr.get_or_compile_kernel("optim", "muon_param_update")

            for i, (param, grad) in enumerate(self.params):
                if grad is None:
                    continue
                if param.dtype == np.float32 and grad.dtype == np.float32:
                    N = param.size
                    p_key = id(param)
                    if p_key not in self._gpu_buffers:
                        buf_p = dev.create_buffer(data=np.ascontiguousarray(param), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        buf_g = dev.create_buffer(data=np.ascontiguousarray(grad), usage=slangpy.BufferUsage.shader_resource)
                        buf_v = dev.create_buffer(data=np.ascontiguousarray(self.v[i]), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        buf_u = dev.create_buffer(size=N * 4, usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        self._gpu_buffers[p_key] = (buf_p, buf_g, buf_v, buf_u)
                    else:
                        buf_p, buf_g, buf_v, buf_u = self._gpu_buffers[p_key]
                        buf_g.copy_from_numpy(np.ascontiguousarray(grad))

                    # Momentum and Nesterov update on GPU
                    k_mom.dispatch(
                        thread_count=[N, 1, 1],
                        param=buf_p,
                        grad=buf_g,
                        v=buf_v,
                        update=buf_u,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay,
                        nesterov=1 if self.nesterov else 0,
                        n_elems=N
                    )
                    dev.wait_for_idle()

                    # Orthogonalize 2D weight matrices via Newton-Schulz
                    u_arr = buf_u.to_numpy().view(np.float32)[:N].reshape(param.shape)
                    step_lr = self.lr
                    if u_arr.ndim >= 2:
                        orig_shape = u_arr.shape
                        mat = u_arr.reshape(orig_shape[0], -1) if u_arr.ndim > 2 else u_arr
                        mat_ortho = _zeropower_via_newtonschulz5(mat, steps=self.ns_steps)
                        update = mat_ortho.reshape(orig_shape)
                        aspect = max(1.0, float(mat.shape[0]) / float(mat.shape[1])) ** 0.5
                        step_lr = step_lr * aspect
                    else:
                        update = u_arr

                    # Parameter update on GPU
                    buf_ortho = dev.create_buffer(data=np.ascontiguousarray(update, dtype=np.float32), usage=slangpy.BufferUsage.shader_resource)
                    k_param.dispatch(
                        thread_count=[N, 1, 1],
                        param=buf_p,
                        ortho=buf_ortho,
                        step_lr=step_lr,
                        n_elems=N
                    )
                    dev.wait_for_idle()
                    param[:] = buf_p.to_numpy().view(np.float32)[:N].reshape(param.shape)
                    continue

        # CPU Fallback
        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            g = grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * param

            v = self.v[i]
            v[:] = self.momentum * v + (1.0 - self.momentum) * g

            if self.nesterov:
                update = (1.0 - self.momentum) * g + self.momentum * v
            else:
                update = v.copy()

            step_lr = self.lr

            if update.ndim >= 2:
                orig_shape = update.shape
                mat = update.reshape(orig_shape[0], -1) if update.ndim > 2 else update
                mat_ortho = _zeropower_via_newtonschulz5(mat, steps=self.ns_steps)
                update = mat_ortho.reshape(orig_shape)
                aspect = max(1.0, float(mat.shape[0]) / float(mat.shape[1])) ** 0.5
                step_lr = step_lr * aspect

            param -= step_lr * update


class RMSprop(Optimizer):
    """
    RMSprop optimizer (Hinton, Coursera lecture 6).
    Accelerated with Slang compute shaders.
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        use_gpu: bool = True,
    ):
        super().__init__(params, lr, use_gpu=use_gpu)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)

        self.v = [np.zeros_like(p) for p, _ in self.params]
        self.buf = [np.zeros_like(p) if momentum > 0.0 else None for p, _ in self.params]

    def step(self) -> None:
        mgr = None
        if self.use_gpu:
            try:
                from slang_kans.device import get_device
                mgr = get_device()
                if not mgr.is_gpu_available():
                    mgr = None
            except Exception:
                mgr = None

        if mgr is not None:
            dev = mgr.device
            kernel = mgr.get_or_compile_kernel("optim", "rmsprop_step")
            for i, (param, grad) in enumerate(self.params):
                if grad is None:
                    continue
                if param.dtype == np.float32 and grad.dtype == np.float32:
                    N = param.size
                    p_key = id(param)
                    has_mom = 1 if self.momentum > 0.0 else 0
                    buf_arr = self.buf[i]
                    if buf_arr is None:
                        buf_arr = np.zeros_like(param)
                        self.buf[i] = buf_arr

                    if p_key not in self._gpu_buffers:
                        buf_p = dev.create_buffer(data=np.ascontiguousarray(param), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        buf_g = dev.create_buffer(data=np.ascontiguousarray(grad), usage=slangpy.BufferUsage.shader_resource)
                        buf_v = dev.create_buffer(data=np.ascontiguousarray(self.v[i]), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        buf_b = dev.create_buffer(data=np.ascontiguousarray(buf_arr), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        self._gpu_buffers[p_key] = (buf_p, buf_g, buf_v, buf_b)
                    else:
                        buf_p, buf_g, buf_v, buf_b = self._gpu_buffers[p_key]
                        buf_g.copy_from_numpy(np.ascontiguousarray(grad))

                    kernel.dispatch(
                        thread_count=[N, 1, 1],
                        param=buf_p,
                        grad=buf_g,
                        v=buf_v,
                        buf=buf_b,
                        lr=self.lr,
                        alpha=self.alpha,
                        eps=self.eps,
                        weight_decay=self.weight_decay,
                        momentum=self.momentum,
                        has_momentum=has_mom,
                        n_elems=N
                    )
                    dev.wait_for_idle()
                    param[:] = buf_p.to_numpy().view(np.float32)[:N].reshape(param.shape)
                    continue

        # CPU Fallback
        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            g = grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * param

            v = self.v[i]
            v[:] = self.alpha * v + (1.0 - self.alpha) * (g * g)

            avg = g / (np.sqrt(v) + self.eps)

            if self.momentum > 0.0:
                buf = self.buf[i]
                if buf is None:
                    buf = np.zeros_like(param)
                    self.buf[i] = buf
                buf[:] = self.momentum * buf + avg
                param -= self.lr * buf
            else:
                param -= self.lr * avg


class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum) optimizer (Chen et al., Google Brain, 2023).
    Tracks momentum and uses only the sign of updates, yielding high memory efficiency.
    Accelerated with Slang compute shaders.
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_gpu: bool = True,
    ):
        super().__init__(params, lr, use_gpu=use_gpu)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.weight_decay = float(weight_decay)
        self.m = [np.zeros_like(p) for p, _ in self.params]

    def step(self) -> None:
        mgr = None
        if self.use_gpu:
            try:
                from slang_kans.device import get_device
                mgr = get_device()
                if not mgr.is_gpu_available():
                    mgr = None
            except Exception:
                mgr = None

        if mgr is not None:
            dev = mgr.device
            kernel = mgr.get_or_compile_kernel("optim", "lion_step")
            for i, (param, grad) in enumerate(self.params):
                if grad is None:
                    continue
                if param.dtype == np.float32 and grad.dtype == np.float32:
                    N = param.size
                    p_key = id(param)
                    if p_key not in self._gpu_buffers:
                        buf_p = dev.create_buffer(data=np.ascontiguousarray(param), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        buf_g = dev.create_buffer(data=np.ascontiguousarray(grad), usage=slangpy.BufferUsage.shader_resource)
                        buf_m = dev.create_buffer(data=np.ascontiguousarray(self.m[i]), usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                        self._gpu_buffers[p_key] = (buf_p, buf_g, buf_m)
                    else:
                        buf_p, buf_g, buf_m, buf_v = self._gpu_buffers[p_key] if len(self._gpu_buffers[p_key]) == 4 else (*self._gpu_buffers[p_key], None)
                        buf_g.copy_from_numpy(np.ascontiguousarray(grad))

                    kernel.dispatch(
                        thread_count=[N, 1, 1],
                        param=buf_p,
                        grad=buf_g,
                        m=buf_m,
                        lr=self.lr,
                        beta1=self.beta1,
                        beta2=self.beta2,
                        weight_decay=self.weight_decay,
                        n_elems=N
                    )
                    dev.wait_for_idle()
                    param[:] = buf_p.to_numpy().view(np.float32)[:N].reshape(param.shape)
                    continue

        # CPU Fallback
        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            if self.weight_decay != 0.0:
                param -= self.lr * self.weight_decay * param

            g = grad
            m = self.m[i]

            update = np.sign(self.beta1 * m + (1.0 - self.beta1) * g)
            param -= self.lr * update

            m[:] = self.beta2 * m + (1.0 - self.beta2) * g
