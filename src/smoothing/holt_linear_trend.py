import sys
from pathlib import Path
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt


root_path = Path().resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))


def load_time_series_from_csv(
    file_path: str,
    skiprows: range,
    nrows: int,
    time_col: str = "msur_dt",
    value_col: str = "inflow_flux_tot",
    freq: str = "min",
) -> pd.Series:
    df = pd.read_csv(file_path, skiprows=skiprows, nrows=nrows)
    df[time_col] = pd.to_datetime(df[time_col])
    return pd.Series(
        df[value_col].values, index=pd.DatetimeIndex(df[time_col], freq=freq)
    )


series = load_time_series_from_csv(
    file_path=r"D:\dev\modules\pt_eh_inflow_data.csv",
    skiprows=range(1, 14401 + (1440 * 7) * 2),
    nrows=1440 * 7,
    time_col="msur_dt",
    value_col="inflow_flux_tot",
    freq="min",
)

# 시계열 데이터 준비
data = series.dropna().astype(float)

# Holt 모델 학습
model = Holt(data)
fit = model.fit(smoothing_level=0.8, smoothing_trend=0.2)

# 예측
steps = 5
forecast = fit.forecast(steps)

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(data.index, data.values, label="Observed")
plt.plot(data.index, fit.fittedvalues, label="Fitted")
plt.plot(forecast.index, forecast.values, label="Forecast", linestyle="--")
plt.legend()
plt.title("Holt's Linear Trend (이중 지수 평활화)")
plt.tight_layout()
plt.show()
