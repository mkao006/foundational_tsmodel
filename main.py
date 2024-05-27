import json
import os
import warnings

from datasetsforecast.m4 import M4

from src.evaluator import Evaluator, smape_metric
from model_registry.chronos import chronosFMSmall, chronosFMLarge
from model_registry.nixtla_sfmodel import sf_models
from model_registry.nixtla_nfmodel import nf_models

warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DATASET_DIRECTORY = "datasets"
GROUP = "Monthly"
HORIZON = 12

# Get the data
m4_datasets = M4.load(directory=DATASET_DIRECTORY, group=GROUP)[0]

# Run benchmark
evaluator = Evaluator(
    data=m4_datasets,
    models=[
        chronosFMSmall,
        # chronosFMLarge,
    ]
    + sf_models
    + nf_models,
    metric=smape_metric,
    sample_size=50,
    train_window_size=24,
    test_window_size=12,
)

result = evaluator.evaluate()
print(json.dumps(result, indent=4))
