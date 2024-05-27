from src.model import NixtlaModel, ForecastParam
from statsforecast.models import AutoARIMA
from statsforecast.models import CrostonClassic as Croston
from statsforecast.models import DynamicOptimizedTheta as DOT
from statsforecast.models import HistoricAverage, SeasonalNaive

HORIZON = 12
# Initiate statstical models
#
# NOTE (MICHAEL): Somehow loading the statsforecast model package causes the
#                 Chronos model to fail with segmentation fault 11 error.

nixtla_stats_model_params = {
    "training_params": {"freq": "M"},
    "prediction_params": {"h": HORIZON},
}

sf_models = [
    NixtlaModel(m, ForecastParam(**nixtla_stats_model_params))
    for m in [
        AutoARIMA(season_length=HORIZON),
        Croston(),
        SeasonalNaive(season_length=HORIZON),
        HistoricAverage(),
        DOT(season_length=HORIZON),
    ]
]
