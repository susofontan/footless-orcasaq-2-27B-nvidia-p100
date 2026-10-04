"""A ctypes bridge to the CUDA driver API for the OrcaSAQ-2-27B sm_60 runtime.

The runtime talks to the GPU through this, with no framework underneath: the
driver library is loaded directly, kernels are compiled from source by nvcc at
open time, and every launch is a `cuLaunchKernel` call.

Why the driver API and not the runtime API: `cuLaunchKernel` takes a function
handle obtained from a loaded module, which is what compiling a `.cu` to a
`.cubin` and loading it gives us. The runtime API would need a host function
symbol this process never links against.

Everything here is sm_60 (Pascal/GP100). CUDA 12.6 is the last toolkit that
compiles for it; 13.0 refuses, so the compiler path is pinned.

Adapted from the Xing package's bridge
(`models/Xing4.0-29B-A4B/footless/cuda_sm60/bridge.py`), which measured the
copy below on this machine. The one structural difference is that this runtime
has two kernel sources, so compilation and loading are per source and kernel
handles are looked up per module:

    dev  = Device()
    arch = dev.load_module(dev.compile(HERE / "kernels.cu"), "arch")
    exl3 = dev.load_module(dev.compile(HERE / "kernels_exl3.cu"), "exl3")
    fn   = arch.kernel("rmsnorm_f16")      # from either module, by its own name

A kernel name is not unique across modules: `Module.kernel` caches per module,
because one dict keyed by name alone would hand back the other source's code
for a name both files define.
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

NVCC = os.environ.get("FOOTLESS_NVCC", "/usr/local/cuda-12.6/bin/nvcc")
ARCH = "sm_60"

# ---------------------------------------------------------------- driver types
CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUevent = ctypes.c_void_p
CUdeviceptr = ctypes.c_ulonglong

# CUdevice_attribute values used here (cuda.h)
_DEVICE_ATTR_CC_MAJOR = 75
_DEVICE_ATTR_CC_MINOR = 76
CU_STREAM_NON_BLOCKING = 0x1

_cuda = ctypes.CDLL("libcuda.so.1")


class CudaError(RuntimeError):
    """A driver call that did not return CUDA_SUCCESS."""


def _bind(name: str, restype, argtypes: list) -> ctypes._CFuncPtr:
    fn = getattr(_cuda, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


_cuInit = _bind("cuInit", CUresult, [ctypes.c_uint])
_cuDeviceGet = _bind("cuDeviceGet", CUresult, [ctypes.POINTER(CUdevice), ctypes.c_int])
_cuDeviceGetAttribute = _bind("cuDeviceGetAttribute", CUresult,
                              [ctypes.POINTER(ctypes.c_int), ctypes.c_int, CUdevice])
_cuDeviceGetName = _bind("cuDeviceGetName", CUresult,
                         [ctypes.c_char_p, ctypes.c_int, CUdevice])
_cuDeviceTotalMem = _bind("cuDeviceTotalMem_v2", CUresult,
                          [ctypes.POINTER(ctypes.c_size_t), CUdevice])
_cuCtxCreate = _bind("cuCtxCreate_v2", CUresult,
                     [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice])
_cuCtxDestroy = _bind("cuCtxDestroy_v2", CUresult, [CUcontext])
_cuCtxSynchronize = _bind("cuCtxSynchronize", CUresult, [])
_cuMemAlloc = _bind("cuMemAlloc_v2", CUresult, [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t])
_cuMemFree = _bind("cuMemFree_v2", CUresult, [CUdeviceptr])
_cuMemAllocHost = _bind("cuMemAllocHost_v2", CUresult,
                        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t])
_cuMemFreeHost = _bind("cuMemFreeHost", CUresult, [ctypes.c_void_p])
_cuMemGetInfo = _bind("cuMemGetInfo_v2", CUresult,
                      [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)])
_cuMemcpyHtoD = _bind("cuMemcpyHtoD_v2", CUresult,
                      [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t])
_cuMemcpyDtoH = _bind("cuMemcpyDtoH_v2", CUresult,
                      [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t])
_cuMemcpyDtoD = _bind("cuMemcpyDtoD_v2", CUresult,
                      [CUdeviceptr, CUdeviceptr, ctypes.c_size_t])
_cuMemcpyDtoHAsync = _bind("cuMemcpyDtoHAsync_v2", CUresult,
                           [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t, CUstream])
_cuMemcpyHtoDAsync = _bind("cuMemcpyHtoDAsync_v2", CUresult,
                           [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t, CUstream])
_cuMemcpyDtoDAsync = _bind("cuMemcpyDtoDAsync_v2", CUresult,
                           [CUdeviceptr, CUdeviceptr, ctypes.c_size_t, CUstream])
_cuModuleLoad = _bind("cuModuleLoad", CUresult, [ctypes.POINTER(CUmodule), ctypes.c_char_p])
_cuModuleGetFunction = _bind("cuModuleGetFunction", CUresult,
                             [ctypes.POINTER(CUfunction), CUmodule, ctypes.c_char_p])
_cuLaunchKernel = _bind("cuLaunchKernel", CUresult,
                        [CUfunction, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                         ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                         ctypes.c_uint, CUstream, ctypes.POINTER(ctypes.c_void_p),
                         ctypes.POINTER(ctypes.c_void_p)])
_cuStreamCreate = _bind("cuStreamCreate", CUresult,
                        [ctypes.POINTER(CUstream), ctypes.c_uint])
_cuStreamSynchronize = _bind("cuStreamSynchronize", CUresult, [CUstream])
_cuStreamDestroy = _bind("cuStreamDestroy_v2", CUresult, [CUstream])
_cuEventCreate = _bind("cuEventCreate", CUresult, [ctypes.POINTER(CUevent), ctypes.c_uint])
_cuEventRecord = _bind("cuEventRecord", CUresult, [CUevent, CUstream])
_cuEventSynchronize = _bind("cuEventSynchronize", CUresult, [CUevent])
_cuEventElapsedTime = _bind("cuEventElapsedTime", CUresult,
                            [ctypes.POINTER(ctypes.c_float), CUevent, CUevent])
_cuEventDestroy = _bind("cuEventDestroy_v2", CUresult, [CUevent])
_cuStreamWaitEvent = _bind("cuStreamWaitEvent", CUresult,
                           [CUstream, CUevent, ctypes.c_uint])
# The Xing bridge reported a failing driver call by number only. Naming the
# code is worth one extra binding: a launch that fails with
# CUDA_ERROR_ILLEGAL_ADDRESS reads very differently from one that fails with
# CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, and the number alone does not say which.
_cuGetErrorName = _bind("cuGetErrorName", CUresult,
                        [CUresult, ctypes.POINTER(ctypes.c_char_p)])


def _error_name(code: int) -> str:
    out = ctypes.c_char_p()
    if _cuGetErrorName(code, ctypes.byref(out)) == 0 and out.value:
        return out.value.decode()
    return ""


def _check(result: int, what: str) -> None:
    if result != 0:
        name = _error_name(result)
        raise CudaError(f"{what} failed with CUresult {result}"
                        + (f" ({name})" if name else ""))


# ---------------------------------------------------------------- launch args
# cuLaunchKernel takes an array of POINTERS to the parameter values, so the
# obvious way to build it — one ctypes object per argument, one cast per
# argument, a fresh array per launch — cost ~8 us of every dispatch when it was
# measured here, against a 3.3 us driver call. The block below holds the values
# in one fixed buffer of 8-byte slots with the pointer array built once, so a
# launch is one store per argument plus the driver call. Slot `i` is prealigned
# to 8 bytes, which satisfies every parameter type this package passes.
_ARGS_LIMIT = 16
_pack_u64 = struct.Struct("<Q").pack_into
_pack_f32 = struct.Struct("<f").pack_into


class _ParamBlock:
    """One reused parameter block: values in place, pointers built once."""

    __slots__ = ("buf", "arr", "slots")

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.buf = ctypes.create_string_buffer(8 * slots)
        base = ctypes.addressof(self.buf)
        self.arr = (ctypes.c_void_p * slots)(*[base + 8 * i for i in range(slots)])


class DeviceBuffer:
    """Device memory. Freed explicitly; the runtime owns its arenas."""

    __slots__ = ("ptr", "nbytes", "_dev")

    def __init__(self, ptr: int, nbytes: int, dev: "Device") -> None:
        self.ptr = ptr
        self.nbytes = nbytes
        self._dev = dev

    def free(self) -> None:
        if self.ptr:
            _check(_cuMemFree(self.ptr), "cuMemFree")
            self.ptr = 0
            self.nbytes = 0

    def __repr__(self) -> str:
        return f"<DeviceBuffer {self.nbytes} B at 0x{self.ptr:x}>"


class HostBuffer:
    """Page-locked host memory, so a copy to the GPU can be a real DMA."""

    __slots__ = ("_p", "_dev", "nbytes")

    def __init__(self, ptr: int, nbytes: int, dev: "Device") -> None:
        self._p = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
        self.nbytes = nbytes
        self._dev = dev

    def write(self, offset: int, data: bytes) -> None:
        _span(self, offset, len(data), "HostBuffer.write")
        ctypes.memmove(ctypes.addressof(self._p.contents) + offset, data, len(data))

    def read(self, offset: int, nbytes: int) -> bytes:
        _span(self, offset, nbytes, "HostBuffer.read")
        return ctypes.string_at(ctypes.addressof(self._p.contents) + offset, nbytes)

    def free(self) -> None:
        if self._p:
            _check(_cuMemFreeHost(ctypes.cast(self._p, ctypes.c_void_p)), "cuMemFreeHost")
            self._p = None
            self.nbytes = 0


class Module:
    """One loaded cubin, and the kernel handles looked up inside it.

    `Device.kernel(module, name)` in the Xing bridge cached by name alone. With
    two sources that is wrong the moment both define a name — it would return
    the first module's handle for the second module's kernel — so the cache
    lives per module and there is no shared name space.
    """

    __slots__ = ("handle", "name", "cubin", "_functions")

    def __init__(self, handle: CUmodule, name: str, cubin: Path) -> None:
        self.handle = handle
        self.name = name
        self.cubin = cubin
        self._functions: dict[str, CUfunction] = {}

    def kernel(self, name: str) -> CUfunction:
        """The handle for `name` in this module, looked up once."""
        fn = self._functions.get(name)
        if fn is None:
            out = CUfunction()
            _check(_cuModuleGetFunction(ctypes.byref(out), self.handle, name.encode()),
                   f"cuModuleGetFunction({self.name}.{name})")
            fn = self._functions[name] = out
        return fn

    def __repr__(self) -> str:
        return f"<Module {self.name} {len(self._functions)} kernels>"


class Device:
    """One CUDA context on one device, plus the modules loaded into it."""

    def __init__(self, ordinal: int = 0, verbose: bool = False) -> None:
        _check(_cuInit(0), "cuInit")
        dev = CUdevice()
        _check(_cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
        ctx = CUcontext()
        _check(_cuCtxCreate(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
        self.context = ctx
        self.device = dev
        self.modules: list[Module] = []
        self._params: _ParamBlock | None = None
        self.name = self._name(dev)
        self.cc = (self._attr(dev, _DEVICE_ATTR_CC_MAJOR),
                   self._attr(dev, _DEVICE_ATTR_CC_MINOR))
        total = ctypes.c_size_t()
        _check(_cuDeviceTotalMem(ctypes.byref(total), dev), "cuDeviceTotalMem")
        self.total_memory = total.value
        if verbose:
            print(f"[cuda] {self.name} sm_{self.cc[0]}{self.cc[1]} "
                  f"{self.total_memory / 2**30:.2f} GiB", file=sys.stderr)

    @staticmethod
    def _attr(dev: CUdevice, attr: int) -> int:
        out = ctypes.c_int()
        _check(_cuDeviceGetAttribute(ctypes.byref(out), attr, dev), "cuDeviceGetAttribute")
        return out.value

    @staticmethod
    def _name(dev: CUdevice) -> str:
        buf = ctypes.create_string_buffer(128)
        _check(_cuDeviceGetName(buf, 128, dev), "cuDeviceGetName")
        return buf.value.decode()

    # ------------------------------------------------------------ compilation
    # A static method, unlike the Xing bridge's: compiling is an nvcc call and
    # touches no context, so `Device.compile(source)` works with no device
    # present (which is what lets the nvcc failure tests run without a GPU).
    @staticmethod
    def compile(source: Path, cubin: Path | None = None,
                extra_flags: list[str] | None = None) -> Path:
        """nvcc source -> cubin, skipping the work when the cubin is newer.

        One source per call; a runtime with two kernel files calls this twice
        and gets two cubins. The skip compares mtimes only, so it does not
        notice changed `extra_flags` — delete the cubin when flags change.

        Failures are `CudaError`: a missing nvcc, a compiler that cannot be
        executed, and a non-zero exit all say so, and a non-zero exit carries
        the tail of nvcc's stderr (the actual diagnostic). A missing *source*
        is left as the plain FileNotFoundError it is.
        """
        source = Path(source)
        cubin = Path(cubin) if cubin else source.with_suffix(".cubin")
        if cubin.exists() and cubin.stat().st_size > 0 \
                and cubin.stat().st_mtime >= source.stat().st_mtime:
            return cubin
        cmd = [NVCC, "-cubin", f"-arch={ARCH}", "-O3", "--use_fast_math",
               "-o", str(cubin), str(source)]
        cmd += extra_flags or []
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as exc:
            raise CudaError(f"cannot run nvcc {NVCC} "
                            f"(set FOOTLESS_NVCC): {exc}") from exc
        if proc.returncode:
            raise CudaError(f"nvcc failed for {source}:\n{proc.stderr[-4000:]}")
        if proc.stderr.strip():
            print(f"[nvcc] {proc.stderr.strip()[:500]}", file=sys.stderr)
        print(f"[cuda] compiled {source.name} in {time.time() - t0:.1f}s", file=sys.stderr)
        return cubin

    def load_module(self, cubin: Path, name: str | None = None) -> Module:
        """Load a cubin as a module; `name` labels it in errors and reprs."""
        cubin = Path(cubin)
        handle = CUmodule()
        _check(_cuModuleLoad(ctypes.byref(handle), str(cubin).encode()),
               f"cuModuleLoad({cubin.name})")
        module = Module(handle, name or cubin.stem, cubin)
        self.modules.append(module)
        return module

    # ------------------------------------------------------------ memory
    def alloc(self, nbytes: int) -> DeviceBuffer:
        if nbytes <= 0:
            raise ValueError(f"allocating {nbytes} bytes")
        ptr = CUdeviceptr()
        _check(_cuMemAlloc(ctypes.byref(ptr), nbytes), f"cuMemAlloc({nbytes})")
        return DeviceBuffer(ptr.value, nbytes, self)

    def alloc_host(self, nbytes: int) -> HostBuffer:
        ptr = ctypes.c_void_p()
        _check(_cuMemAllocHost(ctypes.byref(ptr), nbytes), f"cuMemAllocHost({nbytes})")
        return HostBuffer(ptr.value, nbytes, self)

    def free_memory(self) -> tuple[int, int]:
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        _check(_cuMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
        return free.value, total.value

    def htod(self, dst: DeviceBuffer, data, offset: int = 0, stream=None) -> None:
        buf, n = _as_buffer(data)
        _span(dst, offset, n, "htod")
        if stream:
            _check(_cuMemcpyHtoDAsync(dst.ptr + offset, buf, n, stream), "cuMemcpyHtoDAsync")
        else:
            _check(_cuMemcpyHtoD(dst.ptr + offset, buf, n), "cuMemcpyHtoD")

    def htod_async(self, dst: DeviceBuffer, data, offset: int = 0, stream=None) -> None:
        """HtoD ordered on the default stream, without the host waiting for the
        device. A synchronous HtoD of a few bytes is not cheap here: it is
        ordered against all queued work, so it costs a device round trip per
        call, which is 38 of them a token."""
        buf, n = _as_buffer(data)
        _span(dst, offset, n, "htod_async")
        _check(_cuMemcpyHtoDAsync(dst.ptr + offset, buf, n, stream), "cuMemcpyHtoDAsync")

    def dtoh(self, src: DeviceBuffer, nbytes: int | None = None, offset: int = 0) -> bytes:
        n = src.nbytes - offset if nbytes is None else nbytes
        _span(src, offset, n, "dtoh")
        out = ctypes.create_string_buffer(n)
        _check(_cuMemcpyDtoH(out, src.ptr + offset, n), "cuMemcpyDtoH")
        return out.raw

    def dtoh_async(self, dst: HostBuffer, src: DeviceBuffer, nbytes: int,
                   offset: int = 0, dst_offset: int = 0, stream=None) -> None:
        """Into page-locked memory on `stream`, so the host can wait on an event
        for just this copy instead of on the whole device."""
        _span(src, offset, nbytes, "dtoh_async")
        _span(dst, dst_offset, nbytes, "dtoh_async")
        _check(_cuMemcpyDtoHAsync(
            ctypes.c_void_p(ctypes.addressof(dst._p.contents) + dst_offset),
            src.ptr + offset, nbytes, stream), "cuMemcpyDtoHAsync")

    def dtod(self, dst: DeviceBuffer, src: DeviceBuffer, nbytes: int,
             dst_off: int = 0, src_off: int = 0) -> None:
        _span(dst, dst_off, nbytes, "dtod")
        _span(src, src_off, nbytes, "dtod")
        _check(_cuMemcpyDtoD(dst.ptr + dst_off, src.ptr + src_off, nbytes), "cuMemcpyDtoD")

    def dtod_async(self, dst: DeviceBuffer, src: DeviceBuffer, nbytes: int,
                   dst_off: int = 0, src_off: int = 0, stream=None) -> None:
        _span(dst, dst_off, nbytes, "dtod_async")
        _span(src, src_off, nbytes, "dtod_async")
        _check(_cuMemcpyDtoDAsync(dst.ptr + dst_off, src.ptr + src_off, nbytes, stream),
               "cuMemcpyDtoDAsync")

    # ------------------------------------------------------------ streams
    def stream(self, non_blocking: bool = True):
        """A stream for the copies and the launches that must overlap.

        Non-blocking by default, deliberately: a non-blocking stream does not
        serialise against the legacy default stream, which is what lets a DMA
        on this stream run beside compute on stream 0. Ordering it to compute
        is then the caller's job, with `record` / `wait` — not with a device
        wide sync.
        """
        s = CUstream()
        flags = CU_STREAM_NON_BLOCKING if non_blocking else 0
        _check(_cuStreamCreate(ctypes.byref(s), flags), "cuStreamCreate")
        return s

    def stream_sync(self, s) -> None:
        _check(_cuStreamSynchronize(s), "cuStreamSynchronize")

    def stream_free(self, s) -> None:
        _check(_cuStreamDestroy(s), "cuStreamDestroy")

    # ------------------------------------------------------------ events
    def event(self):
        e = CUevent()
        _check(_cuEventCreate(ctypes.byref(e), 0), "cuEventCreate")
        return e

    def record(self, event, stream=None) -> None:
        """Mark where the work enqueued on `stream` has got to."""
        _check(_cuEventRecord(event, stream), "cuEventRecord")

    def wait(self, stream, event) -> None:
        """Make `stream` (the legacy default stream when None) wait for `event`."""
        _check(_cuStreamWaitEvent(stream, event, 0), "cuStreamWaitEvent")

    def event_sync(self, event) -> None:
        _check(_cuEventSynchronize(event), "cuEventSynchronize")

    def elapsed_ms(self, first, second) -> float:
        """Device milliseconds between two events on the same stream: the wall
        the GPU took to drain from one mark to the other, idle time included."""
        out = ctypes.c_float()
        _check(_cuEventElapsedTime(ctypes.byref(out), first, second),
               "cuEventElapsedTime")
        return out.value

    def event_free(self, event) -> None:
        _check(_cuEventDestroy(event), "cuEventDestroy")

    # ------------------------------------------------------------ launch
    def launch(self, fn: CUfunction, grid, block, args=(), shared: int = 0,
               stream=None) -> None:
        """One kernel launch. `args` are the parameter values, in kernel order.

        Plain ints and floats take the fast path: each is stored into a slot of
        one reused parameter block and the driver call is the only ctypes call
        left. Anything else (a ctypes value, an over-long list) falls back to
        the boxed path, which builds a pointer array the way this always did.
        The driver copies the parameters at the call, so the block is free to be
        reused by the next launch.

        `bool` is packed as an integer here rather than sent to the boxed path,
        which ctypes cannot represent. Ints are masked to 64 bits, so a
        negative int wraps the way an `int` parameter expects; an int wider
        than 64 bits is silently truncated (`1 << 70` becomes 0), which no
        parameter this package passes can be.
        """
        n = len(args)
        params = self._params
        if params is None or params.slots < n:
            params = self._params = _ParamBlock(max(n, _ARGS_LIMIT))
        buf = params.buf
        for i, a in enumerate(args):
            t = type(a)
            if t is int:
                _pack_u64(buf, 8 * i, a & 0xFFFFFFFFFFFFFFFF)
            elif t is float:
                # the low four bytes of the slot; the kernel reads a 4-byte
                # float and never sees the stale half
                _pack_f32(buf, 8 * i, a)
            elif t is bool:
                _pack_u64(buf, 8 * i, 1 if a else 0)
            else:
                return self._launch_boxed(fn, grid, block, args, shared, stream)
        gx, gy, gz = _as3(grid)
        bx, by, bz = _as3(block)
        _check(_cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, shared, stream,
                               params.arr, None), "cuLaunchKernel")

    def _launch_boxed(self, fn, grid, block, args, shared: int = 0,
                      stream=None) -> None:
        """The original form: one ctypes object per argument, cast per launch.

        The Xing bridge sent the whole argument list here as soon as one
        argument was not a plain int or float, and ctypes cannot point at a
        Python int — so a mixed list (a ctypes scalar beside a plain int
        pointer) failed outright. Plain ints and floats are boxed here too,
        with the same widths the fast path uses, and a list reaches this path
        only for something ctypes has storage for (an array, a c_void_p, a
        struct).
        """
        keep = []
        for i, a in enumerate(args):
            t = type(a)
            if t is int or t is bool:
                boxed = ctypes.c_ulonglong(int(a) & 0xFFFFFFFFFFFFFFFF)
            elif t is float:
                boxed = ctypes.c_float(a)
            else:
                boxed = a
            try:
                keep.append(ctypes.cast(ctypes.pointer(boxed), ctypes.c_void_p))
            except TypeError as exc:
                raise TypeError(
                    f"launch argument {i} is {type(a).__name__}, which has no "
                    f"storage for a kernel parameter; pass an int, a float, or "
                    f"a ctypes value") from exc
        arr = (ctypes.c_void_p * len(keep))(*keep) if keep else None
        gx, gy, gz = _as3(grid)
        bx, by, bz = _as3(block)
        _check(_cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, shared, stream, arr, None),
               "cuLaunchKernel")

    def sync(self) -> None:
        _check(_cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self) -> None:
        if self.context:
            _check(_cuCtxDestroy(self.context), "cuCtxDestroy")
            self.context = None

    # ------------------------------------------------------------ helpers
    # Kernel-parameter boxing. These return plain ints and floats so a launch
    # rides the fast path above; `ptr` is an int for the same reason.
    def i32(self, v: int) -> int:
        return int(v)

    def u32(self, v: int) -> int:
        return int(v)

    def u64(self, v: int) -> int:
        return int(v)

    def f32(self, v: float) -> float:
        return float(v)

    def ptr(self, buf: DeviceBuffer | HostBuffer) -> int:
        if isinstance(buf, DeviceBuffer):
            return buf.ptr
        if isinstance(buf, HostBuffer):
            return ctypes.addressof(buf._p.contents)
        raise TypeError(f"no device address for {type(buf)}")


def _as_buffer(data) -> tuple[ctypes.c_void_p, int]:
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
        return ctypes.c_char_p(raw), len(raw)
    if isinstance(data, HostBuffer):
        return ctypes.c_void_p(ctypes.addressof(data._p.contents)), data.nbytes
    raise TypeError(f"cannot copy from {type(data)}")


def _span(buf, offset: int, nbytes: int, what: str) -> int:
    """Check one copy against the buffer's own length. The Xing bridge checked
    the htod direction only; every direction is checked here, because an
    overrun on the device is a corrupted neighbour, not an exception."""
    if offset < 0 or nbytes < 0 or offset + nbytes > buf.nbytes:
        raise ValueError(
            f"{what}: {nbytes} B at offset {offset} exceeds {buf.nbytes} B buffer")
    return nbytes


def _as3(v) -> tuple[int, int, int]:
    if isinstance(v, int):
        return v, 1, 1
    a = list(v) + [1, 1]
    return int(a[0]), int(a[1]), int(a[2])
