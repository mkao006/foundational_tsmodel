import time
from typing import Callable
from darts.timeseries import TimeSeries


def timeit(f: Callable):
    """Utility decorator to time a function/method."""

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(f"func: {f.__qualname__} took: {te - ts:.4f} sec")
        return result

    return timed


def darts_metric_converter(f: Callable):
    """Utility function to convert Darts metrics"""

    def adapted(actual, predicted):
        return f(TimeSeries.from_series(actual), TimeSeries.from_series(predicted))

    return adapted
