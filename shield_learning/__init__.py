"""Learning adapter; the SHIELD implementation remains in smrithi_ws."""

def __getattr__(name):
    if name == "ShieldInterceptionEnv":
        from .env import ShieldInterceptionEnv
        return ShieldInterceptionEnv
    raise AttributeError(name)
