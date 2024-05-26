import json

import pandas as pd
from darts.metrics.metrics import smape
from darts.timeseries import TimeSeries
from datasetsforecast.m4 import M4

from evaluator import Evaluator
from model import ChronosFoundationalModel, ForecastParam

DATASET_DIRECTORY = "datasets"
GROUP = "Monthly"

# Get the data
m4_datasets = M4.load(directory=DATASET_DIRECTORY, group=GROUP)[0]

chronos_params = {
    "training_params": {},
    "prediction_params": {"prediction_length": 12, "num_samples": 20},
}

chronosFMSmall = ChronosFoundationalModel(
    pretrained_model_name_or_path="amazon/chronos-t5-small",
    params=ForecastParam(**chronos_params),
)


chronosFMLarge = ChronosFoundationalModel(
    pretrained_model_name_or_path="amazon/chronos-t5-large",
    params=ForecastParam(**chronos_params),
)


# TODO (Michael): Write an adaptor decorator for Darts metric.
def smape_metric(actual=pd.Series, predicted=pd.Series) -> float:
    return smape(TimeSeries.from_series(actual), TimeSeries.from_series(predicted))


evaluator = Evaluator(
    data=m4_datasets,
    models=[
        chronosFMSmall,
        chronosFMLarge,
    ],
    metric=smape_metric,
    sample_size=10,
    train_window_size=12,
    test_window_size=12,
)

result = evaluator.evaluate()
print(json.dumps(result, indent=4))
