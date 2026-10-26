"""ctypes adapter to the existing compiled controller (backend environment only)."""
import ctypes
from pathlib import Path

import numpy as np


class Controller:
    def __init__(self, q):
        library = Path(__file__).resolve().parents[1] / "build/libshield_learning_controller.so"
        if not library.exists():
            raise RuntimeError("Run bash scripts/build_native.sh first")
        self.lib = ctypes.CDLL(str(library))
        vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        self.lib.shield_create.argtypes = [vector]
        self.lib.shield_create.restype = ctypes.c_void_p
        self.lib.shield_destroy.argtypes = [ctypes.c_void_p]
        self.lib.shield_command.argtypes = [ctypes.c_void_p, vector, vector, vector, ctypes.c_double]
        self.lib.shield_tick.argtypes = [ctypes.c_void_p, ctypes.c_void_p, vector]
        self.lib.shield_tick.restype = ctypes.c_int
        self.ptr = self.lib.shield_create(np.ascontiguousarray(q, dtype=np.float64))
        self.last_result = None  # ruckig::Result of the latest tick (0 Working, 1 Finished)
        if not self.ptr:
            raise RuntimeError("Could not construct ExistingRuckigController")

    def command(self, result):
        q = np.ascontiguousarray(np.r_[result.position, result.position2], dtype=np.float64)
        self.lib.shield_command(self.ptr, q,
                                np.ascontiguousarray(result.velocity, dtype=np.float64),
                                np.ascontiguousarray(result.acceleration, dtype=np.float64),
                                float(result.duration))

    def tick(self, q):
        """One 4 ms cycle; q=None runs open-loop on Ruckig's own state (no measurement)."""
        velocity = np.zeros(6, dtype=np.float64)
        measured = None if q is None else np.ascontiguousarray(q, dtype=np.float64)
        result = self.lib.shield_tick(self.ptr, None if measured is None else measured.ctypes.data, velocity)
        if result < 0:
            raise RuntimeError(f"ExistingRuckigController returned error {result}")
        if not np.isfinite(velocity).all():
            raise RuntimeError("Non-finite controller output")
        self.last_result = int(result)
        return velocity

    def close(self):
        if self.ptr:
            self.lib.shield_destroy(self.ptr)
            self.ptr = None
