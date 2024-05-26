from abc import ABC, abstractmethod

import pandas as pd
import torch
from chronos import ChronosPipeline
from pydantic import BaseModel

from src.data import TSDataSchema
from src.utils import timeit


class ForecastParam(BaseModel):
    training_params: dict
    prediction_params: dict


class ForecastModel(ABC):
    """Interface for running benchmarking across different packages."""

    def __init__(self, model, params: ForecastParam):
        self.model = model
        self.params = params

    @abstractmethod
    def predict(self, data: pd.DataFrame):
        """Make a forecast.

        :param data: A Pandas data frame following the Nixtla format (unique_id, ds, y).
        :returns: A Pandas data frame with the same format as input.

        """
        pass

    @abstractmethod
    def fit(self):
        pass

    def name(self) -> str:
        pass


class ChronosFoundationalModel(ForecastModel):
    """Implementation of the AWS Chronos foundational model."""

    def __init__(self, pretrained_model_name_or_path: str, params: ForecastParam):
        self.model_name = pretrained_model_name_or_path
        model = ChronosPipeline.from_pretrained(
            pretrained_model_name_or_path,
            # NOTE (Michael): using cpu instead of mps as some dependency are
            #                 not available on apple silicon.
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )
        super().__init__(model, params)

    @timeit
    def predict(self) -> pd.DataFrame:
        unique_ids = self.training_data[TSDataSchema.unique_id].unique()
        prediction_results = []
        for unique_id in unique_ids:
            current_data = self.training_data[
                self.training_data[TSDataSchema.unique_id] == unique_id
            ]
            current_data_max_time = current_data[TSDataSchema.ds].max()
            y = current_data.sort_values(TSDataSchema.ds).reset_index(drop=True)[
                TSDataSchema.y
            ]
            context = torch.tensor(y)
            prediction = (
                self.model.predict(context=context, **self.params.prediction_params)
                .mean(axis=1)
                .tolist()[0]
            )

            prediction_results.append(
                pd.DataFrame(
                    {
                        TSDataSchema.unique_id: unique_id,
                        TSDataSchema.ds: [
                            current_data_max_time + i + 1
                            for i in range(len(prediction))
                        ],
                        TSDataSchema.y: prediction,
                    }
                )
            )
        return pd.concat(prediction_results)

    def fit(self, data):
        """Chronos is a zero-shot foundational model that does not require
        training.

        """
        self.training_data = data

    def name(self) -> str:
        return f"chronos_{self.model_name}"
