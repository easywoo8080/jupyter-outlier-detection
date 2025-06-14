import sys
from pathlib import Path
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

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

df = series.reset_index()
df.columns = ["ds", "y"]

# 인위적으로 이상치 추가 (10번째, 40번째 인덱스 기준)
# df.loc[10, 'y'] += 300  # 급격한 상승
# df.loc[40, 'y'] -= 400  # 급격한 하락

# Prophet 모델 학습
model = Prophet(
    interval_width=0.95,
    daily_seasonality=True,
    weekly_seasonality=False,
    yearly_seasonality=False,
)
model.fit(df)
# 미래 10일치 데이터 생성 및 예측
future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)

# 이상치 탐지 (예측 범위 밖인 경우)
result = df.merge(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left"
)
result["is_anomaly"] = (result["y"] < result["yhat_lower"]) | (
    result["y"] > result["yhat_upper"]
)

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(result["ds"], result["y"], label="Actual")
plt.plot(result["ds"], result["yhat"], label="Forecast", linestyle="--")
plt.fill_between(
    result["ds"], result["yhat_lower"], result["yhat_upper"], color="gray", alpha=0.3
)

# 이상치 강조
anomalies = result[result["is_anomaly"]]
plt.scatter(anomalies["ds"], anomalies["y"], color="red", label="Anomaly", zorder=5)

plt.legend()
plt.title("Prophet - 수위 예측 및 이상치 탐지")
plt.xlabel("날짜")
plt.ylabel("수위")
plt.grid(True)
plt.show()
