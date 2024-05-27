from src.model import ChronosFoundationalModel, ForecastParam

HORIZON = 12

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
