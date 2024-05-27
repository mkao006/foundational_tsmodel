import json
import os
import warnings

from datasetsforecast.m4 import M4

from src.evaluator import Evaluator, smape_metric
from src.model import ChronosFoundationalModel, ForecastParam, NixtlaModel

warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DATASET_DIRECTORY = "datasets"
GROUP = "Monthly"
HORIZON = 12

# Get the data
m4_datasets = M4.load(directory=DATASET_DIRECTORY, group=GROUP)[0]

# Initiate chronos (foundational model)
chronos_params = {
    "training_params": {"freq": HORIZON},
    "prediction_params": {"h": HORIZON, "num_samples": 50},
}

chronosFMSmall = ChronosFoundationalModel(
    pretrained_model_name_or_path="amazon/chronos-t5-small",
    params=ForecastParam(**chronos_params),
)


chronosFMLarge = ChronosFoundationalModel(
    pretrained_model_name_or_path="amazon/chronos-t5-large",
    params=ForecastParam(**chronos_params),
)

# Initiate statstical models
#
# NOTE (MICHAEL): Somehow loading the statsforecast model package causes the
#                 Chronos model to fail with segmentation fault 11 error.
from statsforecast.models import AutoARIMA
from statsforecast.models import CrostonClassic as Croston
from statsforecast.models import DynamicOptimizedTheta as DOT
from statsforecast.models import HistoricAverage, SeasonalNaive

nixtla_stats_model_params = {
    "training_params": {"freq": "M"},
    "prediction_params": {"h": HORIZON},
}

sfModels = [
    NixtlaModel(m, ForecastParam(**nixtla_stats_model_params))
    for m in [
        AutoARIMA(season_length=HORIZON),
        Croston(),
        SeasonalNaive(season_length=HORIZON),
        HistoricAverage(),
        DOT(season_length=HORIZON),
    ]
]

from neuralforecast.auto import AutoNBEATS, AutoTiDE

nixtla_neural_model_params = {
    "training_params": {"freq": 1},
    "prediction_params": {"h": HORIZON},
}

# Initiate NN models
nfModels = [
    NixtlaModel(m, ForecastParam(**nixtla_neural_model_params))
    for m in [
        AutoNBEATS(h=HORIZON),
        AutoTiDE(h=HORIZON),
    ]
]

evaluator = Evaluator(
    data=m4_datasets,
    models=[
        chronosFMSmall,
        # chronosFMLarge,
    ]
    + sfModels
    + nfModels,
    metric=smape_metric,
    sample_size=50,
    train_window_size=24,
    test_window_size=12,
)

result = evaluator.evaluate()
print(json.dumps(result, indent=4))
