from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import torch
from chronos import ChronosPipeline
from mlforecast import MLForecast
from mlforecast.core import _get_model_name
from neuralforecast import NeuralForecast
from neuralforecast.common._base_model import BaseModel as nfbm
from pydantic import BaseModel
from sklearn.base import BaseEstimator as mlbm
from statsforecast import StatsForecast
from statsforecast.models import _TS as sfbm

from src.data import TSDataSchema
from src.utils import timeit


# TODO (Michael): Rethink about the params
class TrainingParams(BaseModel):
    freq: Union[int, str]


class PredictionParams(BaseModel):
    h: int


class ForecastParam(BaseModel):
    training_params: TrainingParams
    prediction_params: PredictionParams


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
                self.model.predict(
                    context=context, prediction_length=self.params.prediction_params.h
                )
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


class NixtlaModel(ForecastModel):
    def __init__(self, model: Union[sfbm, mlbm, nfbm], params: ForecastParam):
        model_instance = self._instantiate_model(model=model, params=params)
        super().__init__(model_instance, params)

    @timeit
    def predict(self) -> pd.DataFrame:
        if self.model_type == "sfbm":
            prediction = (
                self.model.forecast(
                    df=self.training_data, h=self.params.prediction_params.h
                )
                .reset_index()
                .rename(columns={self.name(): TSDataSchema.y})
            )
            prediction[TSDataSchema.ds] = self._revert_date_to_int(
                prediction[TSDataSchema.ds]
            )
            return prediction

        elif self.model_type == "mlbm":
            return self.model.predict(
                new_df=self.training_data, h=self.params.prediction_params.h
            )
        else:
            return self.model.predict(df=self.training_data)

    @timeit
    def fit(self, data):
        self.training_data = data
        self.model.fit(self.training_data)

    def name(self) -> str:
        if self.model_type in ("sfbm", "nfbm"):
            return self.model.models[0].__repr__()
        else:
            return _get_model_name(self.model.models[0])

    def _set_model_type(self, model):
        """Set the model type.based on the baseclass of the input model"""
        if isinstance(model, sfbm):
            self.model_type = "sfbm"
        elif isinstance(model, nfbm):
            self.model_type = "nfbm"
        elif isinstance(model, mlbm):
            self.model_type = "mlbm"
        else:
            raise ValueError("model is not a supported type in Nixtla")

    def _instantiate_model(self, model, params: ForecastParam):
        """Instantiate the Nixtla modeul based on the model"""
        self._set_model_type(model)
        if self.model_type == "sfbm":
            return StatsForecast(models=[model], freq=params.training_params.freq)
        elif self.model_type == "mlbm":
            return MLForecast(
                models=[model],
                freq=params.training_params.freq,
                lags=params.training_params.lags,
            )
        else:
            return NeuralForecast(models=[model], freq=params.training_params.freq)

    def _revert_date_to_int(self, s: pd.Series) -> pd.Series:
        """The predict function converts int offset (1, 2, 3) to the
        frequency since epoch time, this function converts it back.

        """
        return [
            (
                d.to_period(self.params.training_params.freq)
                - pd.to_datetime("1970-01-01").to_period(
                    self.params.training_params.freq
                )
            ).n
            + self.training_data[TSDataSchema.ds].max()
            + 1
            for d in s
        ]
