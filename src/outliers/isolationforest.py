import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# 샘플 데이터 생성
rng = np.random.RandomState(42)
normal_data = rng.normal(loc=0, scale=1, size=(100, 2))
anomaly_data = rng.uniform(low=-6, high=6, size=(10, 2))
X = np.vstack([normal_data, anomaly_data])
df = pd.DataFrame(X, columns=["x", "y"])

# 이상치 탐지
model = IsolationForest(contamination=0.1, random_state=42)
df["anomaly"] = model.fit_predict(df[["x", "y"]])

# # 이상치 값을 NaN으로 치환
# df.loc[df["anomaly"] == -1, ["x", "y"]] = np.nan


# # KNN으로 NaN 대체
# imputer = KNNImputer(n_neighbors=5)
# df_imputed = df.copy()
# df_imputed[["x", "y"]] = imputer.fit_transform(df_imputed[["x", "y"]])
0
print(df["anomaly"].value_counts())  # 이상치와 정상치의 개수 출력
# 시각화
plt.figure(figsize=(8, 6))
colors = df["anomaly"].map({1: "blue", -1: "red"})
plt.scatter(df["x"], df["y"], c=colors)
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
