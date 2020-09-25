import time
from functools import wraps

__all__ = [ "timeit" ]

def timeit(logger):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"[{func.__module__}:{func.__name__}] - execution time: {end_time-start_time}")
            return result

        return wrapper

    return decorator

