from src.model import NixtlaModel, ForecastParam
from neuralforecast.auto import AutoNBEATS, AutoTiDE

HORIZON = 12

nixtla_neural_model_params = {
    "training_params": {"freq": 1},
    "prediction_params": {"h": HORIZON},
}

# Initiate NN models
nf_models = [
    NixtlaModel(m, ForecastParam(**nixtla_neural_model_params))
    for m in [
        AutoNBEATS(h=HORIZON),
        AutoTiDE(h=HORIZON),
    ]
]
