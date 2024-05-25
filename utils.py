import time


def timeit(f):
    """Utility decorator to time a function/method."""

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(f"func: {f.__qualname__} took: {te - ts:.4f} sec")
        return result

    return timed
