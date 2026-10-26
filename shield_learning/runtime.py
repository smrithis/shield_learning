"""Local transport/clock for unchanged ROS simulator classes.

Only each imported simulator module's `rospy` reference is replaced. No global
ROS monkeypatch, ROS master, network publishers, or copies of simulation logic.
SimArmBridge.run performs its own integration and command timeout handling.
Its Rate.sleep is a rendezvous: one release executes exactly one native tick.
"""
import importlib.util
import threading
from types import SimpleNamespace

import rospy


def load_simulator(path, runtime):
    spec = importlib.util.spec_from_file_location("shield_local_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.rospy = runtime
    return module


class LocalRuntime:
    def __init__(self, params=None):
        self.params = params or {}
        self.time = 1.0
        self.stopped = False
        self.published = {}
        self.ready = threading.Event()
        self.release = threading.Event()
        self.error = None
        self.Time = SimpleNamespace(now=lambda: rospy.Time.from_sec(self.time))
        self.Duration = rospy.Duration
        self.core = SimpleNamespace(is_initialized=lambda: False)

    def init_node(self, *_args, **_kwargs):
        pass

    def get_param(self, name, *default):
        if name in self.params:
            return self.params[name]
        if default:
            return default[0]
        raise KeyError(name)

    def Publisher(self, topic, *_args, **_kwargs):
        return SimpleNamespace(publish=lambda msg: self.published.__setitem__(topic, msg))

    def Subscriber(self, *_args, **_kwargs):
        return None

    Service = Subscriber
    loginfo = Subscriber
    logwarn_throttle = Subscriber

    def is_shutdown(self):
        return self.stopped

    def Rate(self, hz):
        if hz != 250.0:
            raise ValueError("The controller and bridge must both run at 250 Hz")
        return SimpleNamespace(sleep=self._yield_tick)

    def _yield_tick(self):
        self.ready.set()
        self.release.wait()
        self.release.clear()

    def start(self, bridge):
        def run():
            try:
                bridge.run()
            except BaseException as error:
                self.error = error
                self.ready.set()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self._wait()

    def _wait(self):
        if not self.ready.wait(10):
            raise TimeoutError("SimArmBridge did not finish its tick")
        if self.error:
            raise RuntimeError("SimArmBridge failed") from self.error

    def tick(self, dt):
        self.time += dt
        self.ready.clear()
        self.release.set()
        self._wait()

    def close(self):
        self.stopped = True
        self.release.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("SimArmBridge thread failed to stop")
