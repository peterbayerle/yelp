from .intake import ReviewReader
import time

def time_this(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        output = f(*args, **kwargs)
        time_delta = time.time() - start
        return (output, time_delta) if output else time_delta

    return wrapper
