import warnings
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from darts.metrics.metrics import smape
from darts.timeseries import TimeSeries

from src.data import TSDataSchema
from src.model import ForecastModel


class Evaluator:
    def __init__(
        self,
        data: pd.DataFrame,
        models: List[ForecastModel],
        metric: Callable[pd.Series, pd.Series],
        sample_size: int = 100,
        train_window_size: int = 12,
        test_window_size: int = 12,
    ):
        self.data = TSDataSchema.validate(data)
        self.models = models
        self.metric = metric
        self.sample_size = sample_size
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.sampled_df = None

    def evaluate(self) -> Dict:
        if self.sampled_df is None:
            self._generate_sample()

        result = {}
        for model in self.models:
            cumsum_metric = 0
            for train_set, test_set in self.sampled_df:
                model.fit(train_set)
                prediction = model.predict()[TSDataSchema.y]
                cumsum_metric += self.metric(
                    test_set.reset_index()[TSDataSchema.y], prediction
                )
            average_metric = cumsum_metric / self.sample_size
            result[model.name()] = average_metric
        return result

    def _generate_sample(self):
        uid = self.data[TSDataSchema.unique_id].unique()
        sampled_ids = np.random.choice(uid, self.sample_size, replace=True)
        self.sampled_df = [
            self._split_train_test(
                df=self.data[self.data[TSDataSchema.unique_id] == sample_id],
                train_window_size=self.train_window_size,
                test_window_size=self.test_window_size,
            )
            for sample_id in sampled_ids
        ]

    def _split_train_test(
        self, df: pd.DataFrame, train_window_size: int, test_window_size: int
    ) -> pd.DataFrame:
        if df[TSDataSchema.unique_id].nunique() > 1:
            raise ValueError(
                """ Multiple 'unique_id' found in the data frame, the
            function only supports splitting of univeariate series."""
            )

        ts_duration = df[TSDataSchema.ds].nunique()
        total_window_size = train_window_size + test_window_size
        sampling_space = ts_duration - total_window_size

        if sampling_space < 0:
            warnings.warn(
                """Timeseries has insufficient length, returning empty dataframe"""
            )
            return pd.DataFrame(columns=TSDataSchema.__schema__.columns.keys())
        elif sampling_space == 0:
            return df.iloc[:train_window_size], df.iloc[train_window_size:]
        else:
            window_start = np.random.choice(sampling_space)
            train_end = window_start + train_window_size
            test_end = train_end + test_window_size
            return df.iloc[window_start:train_end], df.iloc[train_end:test_end]


# TODO (Michael): Write an adaptor decorator for Darts metric.
def smape_metric(actual=pd.Series, predicted=pd.Series) -> float:
    return smape(TimeSeries.from_series(actual), TimeSeries.from_series(predicted))
