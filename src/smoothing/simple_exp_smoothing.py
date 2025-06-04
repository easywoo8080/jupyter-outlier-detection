# ses_model.py
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def ses_forecast(series: pd.Series, smoothing_level: float, forecast_periods: int):
    model = SimpleExpSmoothing(series, initialization_method="estimated")
    fit = model.fit(smoothing_level=smoothing_level, optimized=False)
    forecast = fit.forecast(forecast_periods)
    return fit.fittedvalues, forecast

import pandas as pd
from typing import cast

def simple_exponential_smoothing(
    series: pd.Series,
    alpha: float,
    forecast_periods: int
) -> tuple[pd.Series, pd.Series]:
    fitted_values = [series.iloc[0]]
    for t in range(1, len(series)):
        fitted = alpha * series.iloc[t-1] + (1 - alpha) * fitted_values[-1]
        fitted_values.append(fitted)
    fitted_series = pd.Series(fitted_values, index=series.index)
    
    last_level = fitted_values[-1]
    last_timestamp = cast(pd.Timestamp, series.index[-1])
    forecast_start = last_timestamp + pd.offsets.MonthEnd(1)
    forecast_index = pd.date_range(start=forecast_start, periods=forecast_periods, freq='M')
    forecast = pd.Series([last_level] * forecast_periods, index=forecast_index)
    
    return fitted_series, forecast


