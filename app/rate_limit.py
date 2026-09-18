from slowapi import Limiter
from slowapi.util import get_remote_address

# In-memory, per-process limiter -- fine for a single-instance portfolio
# deployment. A multi-instance deployment would need a shared backend
# (e.g. Redis) since each process would otherwise track its own counters.
limiter = Limiter(key_func=get_remote_address)
