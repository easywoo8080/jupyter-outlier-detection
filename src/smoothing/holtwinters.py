import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 예시 시계열 데이터
data = pd.Series(
    [30, 21, 29, 31, 40, 45, 50, 60, 65, 70, 85, 90,
     32, 22, 30, 33, 42, 46, 52, 61, 66, 71, 87, 91]  # 계절성이 있는 데이터
)

# 삼중 지수 평활화 모델 (additive 방식, 계절 주기 12로 가정)
model = ExponentialSmoothing(
    data,
    trend='add',         # 'add' or 'mul' (추세)
    seasonal='add',      # 'add' or 'mul' (계절성)
    seasonal_periods=12  # 계절 주기 (예: 월별이면 12)
)

fit = model.fit()

# 예측 (예: 다음 12개 시점)
forecast = fit.forecast(12)
print(forecast)
