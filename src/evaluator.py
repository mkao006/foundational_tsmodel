import logging
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
        """The valuator function performs benchmark using the data and metrics
        provided for each model.

        """
        self.data = TSDataSchema.validate(data)
        self.models = models
        self.metric = metric
        self.sample_size = sample_size
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.sampled_df = None

    def evaluate(self) -> Dict:
        """Benchmark the model performance"""
        if self.sampled_df is None:
            self._generate_sample()

        result = {}
        for model in self.models:
            logging.info(f"Evaluating model: {model.name()}")
            try:
                cumsum_metric = 0
                for train_set, test_set in self.sampled_df:
                    model.fit(train_set)
                    prediction = model.predict()[TSDataSchema.y]
                    cumsum_metric += self.metric(
                        test_set.reset_index()[TSDataSchema.y], prediction
                    )
                average_metric = cumsum_metric / self.sample_size
                result[model.name()] = average_metric
            except:
                logging.info(f"Evalutation for {model.name()} failed.")
                result[model.name()] = np.nan
        return result

    def _generate_sample(self):
        """Function to generate the cross-validation set. We first sample the
        unique_id which is the ID representing the time series, then for each
        time series, we sample a subset of a time series then split into
        training and test set.

        The output is a list of length n equivalent to the sample size, and
        each element is a tuples which contains the train and test set.

        """
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
        """Given a full time series, sample a subset of the time series and
        split into training and test set.

        """
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
    """Convert the sMAPE from Dart to be compatible with the Evaluator
    interface"""
    return smape(TimeSeries.from_series(actual), TimeSeries.from_series(predicted))
