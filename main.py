import torch
from chronos import ChronosPipeline
from datasetsforecast.m3 import M3
from datasetsforecast.m4 import M4

from model import ChronosFoundationalModel, ForecastParam

DATASET_DIRECTORY = "datasets"
GROUP = "Monthly"

# Get the data
m3_datasets = M3.load(directory=DATASET_DIRECTORY, group=GROUP)[0]
m4_datasets = M4.load(directory=DATASET_DIRECTORY, group=GROUP)[0]

chronos_params = {
    "training_params": {},
    "prediction_params": {"prediction_length": 12, "num_samples": 20},
}

chronosFMSmall = ChronosFoundationalModel(
    ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        # NOTE (Michael): using cpu instead of mps as some dependency are not
        #                 available on apple silicon.
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    ),
    ForecastParam(**chronos_params),
)

cp = chronosFMSmall.predict(m4_datasets)


if __name__ == "__main__":
    pass
