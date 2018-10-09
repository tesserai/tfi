
from tfi.driverbase import Meta as _Meta, import_dir as _import_dir

import pandas as pd

from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics, cross_validation
from fbprophet.plot import plot_cross_validation_metric

import pyarrow
# """
# forecast = m.fit(df).predict(future)
# m = Prophet()
# m.fit(df)
# future = m.make_future_dataframe(periods=365)
# fig1 = m.plot(forecast)
# fig2 = m.plot_components(forecast)

# from fbprophet.diagnostics import performance_metrics
# df_p = performance_metrics(df_cv)

# from fbprophet.plot import plot_cross_validation_metric
# fig = plot_cross_validation_metric(df_cv, metric='mape')

# """

# def add_regressor(
#     self, name, prior_scale=None, standardize='auto', mode=None
# ):
#     """Add an additional regressor to be used for fitting and predicting.

#     The dataframe passed to `fit` and `predict` will have a column with the
#     specified name to be used as a regressor. When standardize='auto', the
#     regressor will be standardized unless it is binary. The regression
#     coefficient is given a prior with the specified scale parameter.
#     Decreasing the prior scale will add additional regularization. If no
#     prior scale is provided, self.holidays_prior_scale will be used.
#     Mode can be specified as either 'additive' or 'multiplicative'. If not
#     specified, self.seasonality_mode will be used. 'additive' means the
#     effect of the regressor will be added to the trend, 'multiplicative'
#     means it will multiply the trend.

#     Parameters
#     ----------
#     name: string name of the regressor.
#     prior_scale: optional float scale for the normal prior. If not
#         provided, self.holidays_prior_scale will be used.
#     standardize: optional, specify whether this regressor will be
#         standardized prior to fitting. Can be 'auto' (standardize if not
#         binary), True, or False.
#     mode: optional, 'additive' or 'multiplicative'. Defaults to
#         self.seasonality_mode.

#     Returns
#     -------
#     The prophet object.
#     """
#     pass

# def add_seasonality(
#     self, name, period, fourier_order, prior_scale=None, mode=None
# ):
#     """Add a seasonal component with specified period, number of Fourier
#     components, and prior scale.

#     Increasing the number of Fourier components allows the seasonality to
#     change more quickly (at risk of overfitting). Default values for yearly
#     and weekly seasonalities are 10 and 3 respectively.

#     Increasing prior scale will allow this seasonality component more
#     flexibility, decreasing will dampen it. If not provided, will use the
#     seasonality_prior_scale provided on Prophet initialization (defaults
#     to 10).

#     Mode can be specified as either 'additive' or 'multiplicative'. If not
#     specified, self.seasonality_mode will be used (defaults to additive).
#     Additive means the seasonality will be added to the trend,
#     multiplicative means it will multiply the trend.

#     Parameters
#     ----------
#     name: string name of the seasonality component.
#     period: float number of days in one period.
#     fourier_order: int number of Fourier components to use.
#     prior_scale: optional float prior scale for this component.
#     mode: optional 'additive' or 'multiplicative'

#     Returns
#     -------
#     The prophet object.
#     """
#     pass


def DataFrame(**column_dtypes):
    columns = column_dtypes.keys()
    dtypes = [column_dtypes[col] for col in columns]
    def _make_dataframe(data):
        print("loading DataFrame from", type(data))
        if isinstance(data, pd.DataFrame):
            df = data[columns]
        elif isinstance(data, dict):
            # Assume it's {"columns": ..., "data": ...}
            df = pd.DataFrame(data=data['data'], columns=data['columns'])
        elif isinstance(data, list):
            # Assume it's [{"col1": ..., "col2": ..., ...}, {"col1{: ..., ...}, ...]
            df = pd.DataFrame.from_records(data)
        elif hasattr(data, 'open') and hasattr(data, 'read_mimetype'):
            mimetype = data.read_mimetype()
            if mimetype == 'text/csv':
                with data.open() as f:
                    df = pd.read_csv(
                        f,
                        header=0,
                        usecols=columns,
                        dtype=column_dtypes,
                    )
            else:
                raise Exception("Expected 'text/csv'; data has unusable mimetype: %s" % mimetype)
        else:
            raise Exception("data is not pandas DataFrame, dict or list and doesn't have both open and read_mimetype methods: %s" % type(data))
        
        print("loaded dataframe with shape", df.shape)
        return df

    d = {}
    _make_dataframe.__getitem__ = lambda ix: d[ix]
    _make_dataframe.get = lambda ix, default: d.get(ix, default)
    return _make_dataframe



DEFAULT_HPARAMS = {
    # 'append_holidays': None,
    'changepoint_prior_scale': 0.05,
    'changepoint_range': 0.8,
    'changepoints': None,
    'growth': 'linear',
    'holidays': None,
    'holidays_prior_scale': 10.0,
    'interval_width': 0.80,
    'mcmc_samples': 0,
    'n_changepoints': 25,
    'seasonality_mode': 'additive',
    'seasonality_prior_scale': 10.0,
    'uncertainty_samples': 1000,
    'seasonalities': [],
    'daily_seasonality': 'auto',
    'weekly_seasonality': 'auto',
    'yearly_seasonality': 'auto',
}

PROPHET_FIELD_CODECS = {
#   'y_scale': (str, np.float64),
#   'changepoints_t': (pyarrow.Array.from_pandas, pyarrow.Array.to_pandas),
#   'start':(str, pd.to_datetime),
  't_scale': (str, pd.to_timedelta),
#   'history': (pyarrow.Table.from_pandas, pyarrow.Table.to_pandas),
#   'history_dates': (pyarrow.Array.from_pandas, pyarrow.Array.to_pandas),
#   'train_component_cols': (pyarrow.Table.from_pandas, pyarrow.Table.to_pandas),
}

def _saved_fields_from_prophet(prophet):
#  {'component_modes': <class 'dict'>,
#   'extra_regressors': <class 'dict'>,
#   'logistic_floor': <class 'bool'>,
#   'params': <class 'dict'>,
#   'seasonalities': <class 'dict'>,
#   'stan_fit': <class 'NoneType'>,
#   'y_scale': <class 'numpy.float64'>,
#   'changepoints_t': <class 'numpy.ndarray'>,
#   'start': <class 'pandas._libs.tslibs.timestamps.Timestamp'>,
#   't_scale': <class 'pandas._libs.tslibs.timedeltas.Timedelta'>,
#   'history': <class 'pandas.core.frame.DataFrame'>,
#   'history_dates': <class 'pandas.core.series.Series'>,
#   'train_component_cols': <class 'pandas.core.frame.DataFrame'>,
#  }

    fields = {
        'changepoints_t': prophet.changepoints_t,
        'component_modes': prophet.component_modes,
        'extra_regressors': prophet.extra_regressors,
        'history': prophet.history,
        'history_dates': prophet.history_dates,
        'logistic_floor': prophet.logistic_floor,
        'params': prophet.params,
        'seasonalities': prophet.seasonalities,
        'stan_fit': prophet.stan_fit,
        'start': prophet.start,
        't_scale': prophet.t_scale,
        'train_component_cols': prophet.train_component_cols,
        'y_scale': prophet.y_scale,
    }
    fields = {
        k: PROPHET_FIELD_CODECS[k][0](v) if k in PROPHET_FIELD_CODECS else v
        for k, v in fields.items()
    }
    print("saved fields", {k: type(v) for k, v in fields.items()})

    return pyarrow.serialize(fields).to_buffer()

def _saved_forecast(forecast):
    return pyarrow.serialize(forecast).to_buffer()

def _new_prophet(default_hparams, seasonalities, saved_fields):
    hparams = {
        k: v if v is not None else DEFAULT_HPARAMS[k]
        for k, v in default_hparams.items()
    }
    m = Prophet(**hparams)

    if seasonalities:
        print("seasonalities", seasonalities)
        for (name, period, fourier_order, prior_scale, mode) in seasonalities:
            m.add_seasonality(name=name, period=period, fourier_order=fourier_order, prior_scale=prior_scale, mode=mode)

    if saved_fields:
        saved_fields = pyarrow.deserialize(saved_fields)
        for k, v in saved_fields.items():
            if k in PROPHET_FIELD_CODECS:
                v = PROPHET_FIELD_CODECS[k][1](v)
            setattr(m, k, v)
    return m

_HistoryDataFrame = DataFrame(ds=str,y=float)


class Model(object, metaclass=_Meta):
  def __init__(self,
        initial_history=None,
        growth=None,
        changepoints=None,
        n_changepoints=None,
        changepoint_range=None,
        yearly_seasonality=None,
        weekly_seasonality=None,
        daily_seasonality=None,
        holidays=None,
        append_holidays=None,
        seasonalities:list=None,
        seasonality_mode:str=None,
        seasonality_prior_scale=None,
        holidays_prior_scale=None,
        changepoint_prior_scale=None,
        mcmc_samples=0,
        interval_width=0.80,
        uncertainty_samples=1000,
        __saved_fields__=None,
        __tfi_tempdirs__=None,
        __tfi_docstrings__=None):
    """
    Parameters
    ----------
    growth: String 'linear' or 'logistic' to specify a linear or logistic
        trend.
    changepoints: List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first `changepoint_range` proportion of the history.
    changepoint_range: Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        `changepoints` is specified.
    Not used if input `changepoints` is supplied.
    yearly_seasonality: Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality: Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    holidays: pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    append_holidays: country name or abbreviation; must be string
    seasonality_mode: 'additive' (default) or 'multiplicative'.
    seasonality_prior_scale: Parameter modulating the strength of the
        seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified
        for individual seasonalities using add_seasonality.
    holidays_prior_scale: Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.
    changepoint_prior_scale: Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
    mcmc_samples: Integer, if greater than 0, will do full Bayesian inference
        with the specified number of MCMC samples. If 0, will do MAP
        estimation.
    interval_width: Float, width of the uncertainty intervals provided
        for the forecast. If mcmc_samples=0, this will be only the uncertainty
        in the trend using the MAP estimate of the extrapolated generative
        model. If mcmc.samples>0, this will be integrated over all model
        parameters, which will include uncertainty in seasonality.
    uncertainty_samples: Number of simulated draws used to estimate
        uncertainty intervals.
    """
    self.__tfi_tempdirs__ = __tfi_tempdirs__
    self.__tfi_docstrings__ = __tfi_docstrings__

    self._initial_history = _HistoryDataFrame(initial_history) if initial_history else None
    self._seasonalities = seasonalities or []

    self._given_default_hparams = {
        'growth': growth,
        'changepoints': changepoints,
        'n_changepoints': n_changepoints,
        'changepoint_range': changepoint_range,
        'yearly_seasonality': yearly_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'daily_seasonality': daily_seasonality,
        'holidays': holidays,
        # 'append_holidays': append_holidays,
        'seasonality_mode': seasonality_mode,
        'seasonality_prior_scale': seasonality_prior_scale,
        'holidays_prior_scale': holidays_prior_scale,
        'changepoint_prior_scale': changepoint_prior_scale,
        'mcmc_samples': mcmc_samples,
        'interval_width': interval_width,
        'uncertainty_samples': uncertainty_samples,
    }

  def __tfi_init__(self):
    pass

  def predict_future(self, history, periods=0, freq='D'):
      return self.predict(history, periods, freq, False)

  def _history(self, additional_history):
    histories = []
    if self._initial_history is not None:
        histories.append(self._initial_history)
    if additional_history is not None:
        histories.append(additional_history)
    if len(histories) == 0:
        raise Exception("Missing history")

    return pd.concat(histories, ignore_index=True)

  def predict(self,
        history: _HistoryDataFrame,
        periods=1,
        freq='D',
        include_history=True):
    """Predict using the prophet model.

    Args:
        history: pd.DataFrame with dates for predictions (column ds), and capacity
            (column cap) if logistic growth. If not provided, predictions are
            made on the history.

    Returns:
        A pd.DataFrame with the forecast components.


    Example Args:
        history: tfi.data.file("tfi://examples/yosemite_temps.csv")
    """
    m = _new_prophet(self._given_default_hparams, self._seasonalities, None)
    h = self._history(_HistoryDataFrame(history))
    m.fit(h)
    
    if periods:
        future = m.make_future_dataframe(periods, freq, include_history)
    else:
        future = history

    forecast = m.predict(future)
    state = {
        'fields': _saved_fields_from_prophet(m),
        'forecast': _saved_forecast(forecast),
    }
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast['ds'] = forecast['ds'].apply(str)
    result = {
        'state': state,
        'forecast': forecast,
    }
    return result

  def cross_validate(self, horizon, state=None, history=None, period=None, initial=None):
    """Cross-Validation for time series.

    Computes forecasts from historical cutoff points. Beginning from
    (end - horizon), works backwards making cutoffs with a spacing of period
    until initial is reached.

    When period is equal to the time interval of the data, this is the
    technique described in https://robjhyndman.com/hyndsight/tscv/ .

    Parameters
    ----------
    model: Prophet class object. Fitted Prophet model
    horizon: string with pd.Timedelta compatible style, e.g., '5 days',
        '3 hours', '10 seconds'.
    period: string with pd.Timedelta compatible style. Simulated forecast will
        be done at every this period. If not provided, 0.5 * horizon is used.
    initial: string with pd.Timedelta compatible style. The first training
        period will begin here. If not provided, 3 * horizon is used.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    if state:
        m = _new_prophet(self._given_default_hparams, [], state['fields'])
    else:
        m = _new_prophet(self._given_default_hparams, self._seasonalities, None)
        m.fit(self._history(_HistoryDataFrame(history)))

    return cross_validation(m, initial=initial, period=period, horizon=horizon)

  def performance_metrics(self, cross_validation):
    """Compute performance metrics from cross-validation results.

    Computes a suite of performance metrics on the output of cross-validation.
    By default the following metrics are included:
    'mse': mean squared error
    'rmse': root mean squared error
    'mae': mean absolute error
    'mape': mean percent error
    'coverage': coverage of the upper and lower intervals

    A subset of these can be specified by passing a list of names as the
    `metrics` argument.

    Metrics are calculated over a rolling window of cross validation
    predictions, after sorting by horizon. The size of that window (number of
    simulated forecast points) is determined by the rolling_window argument,
    which specifies a proportion of simulated forecast points to include in
    each window. rolling_window=0 will compute it separately for each simulated
    forecast point (i.e., 'mse' will actually be squared error with no mean).
    The default of rolling_window=0.1 will use 10% of the rows in df in each
    window. rolling_window=1 will compute the metric across all simulated forecast
    points. The results are set to the right edge of the window.

    The output is a dataframe containing column 'horizon' along with columns
    for each of the metrics computed.

    Parameters
    ----------
    df: The dataframe returned by cross_validation.
    metrics: A list of performance metrics to compute. If not provided, will
        use ['mse', 'rmse', 'mae', 'mape', 'coverage'].
    rolling_window: Proportion of data to use in each rolling window for
        computing the metrics. Should be in [0, 1].

    Returns
    -------
    Dataframe with a column for each metric, and column 'horizon'
    """
    return performance_metrics(cross_validation)

  def plot(self, state):
    fields, forecast = state['fields'], state['forecast']
    m = _new_prophet(self._given_default_hparams, None, fields)
    return m.plot(forecast)

  def plot_components(self, prediction):
    fields, forecast = state['fields'], state['forecast']
    m = _new_prophet(self._given_default_hparams, None, fields)
    return m.plot_components(forecast)

  def plot_cross_validation_metric(self, cross_validation, metric):
    return plot_cross_validation_metric(cross_validation, metric=metric)

# def recent_fraction(keep_threshold: constraint(float, min=0.2, max=1.0), history):
#     if history.shape[0] > 0:
#         cumsum = history.cumsum()
#         total = cumsum.tail(1)['y'][0]
#         drop_threshold = 1 - self.keep_threshold
#         to_drop = cumsum[cumsum['y'] < (total * drop_threshold)]
#         if to_drop.shape[0] > 0:
#             ds_keep_threshold = to_drop.index[0]
#             history = history.ix[history.index >= ds_keep_threshold]
#     return history

# def recency_cliff(history, max_days_back):
#     if max_days_back >= 0 and history.shape[0] > 0:
#         target_date = history.index[-1]
#         horizon = target_date - timedelta(days=max_days_back)
#         history = history[history.index >= horizon]
#     return history


def load(path):
    pass

def dump(model, path):
    pass