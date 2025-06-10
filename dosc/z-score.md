---
title: "Jupyter Notebook 사용법"
tags: [jupyter, python, 데이터분석]
date: 2025-06-02
author: "easywoo8080@github.com"
---


[돌아가기](../README.md)

**MAD (Median Absolute Deviation)** 는 **중앙값 기반 이상치 탐지 방법**입니다. 평균이 아닌 **중앙값**을 사용해 **극단값에 강인한(robust)** 특징이 있습니다.

---

### ✅ MAD 방식 정의

1. **중앙값** 계산

   $$
   \text{Median}(x) 
   $$

2. **중앙값 기준 편차** 계산

   $$
   \text{MAD} = \text{Median}(|x_i - \text{Median}(x)|)
   $$

3. **이상치 판별 기준**
   아래 z-score를 기반으로 이상치를 판단함:

   $$
   z_i = \frac{0.6745 \cdot (x_i - \text{Median}(x))}{\text{MAD}}
   $$

   
\*\*MAD (Median Absolute Deviation)\*\*는 **중앙값 기반 이상치 탐지 방법**입니다. 평균이 아닌 **중앙값**을 사용해 **극단값에 강인한(robust)** 특징이 있습니다.

---

### ✅ MAD 방식 정의

1. **중앙값** 계산

   $$
   \text{Median}(x)
   $$

2. **중앙값 기준 편차** 계산

   $$
   \text{MAD} = \text{Median}(|x_i - \text{Median}(x)|)
   $$

3. **이상치 판별 기준**
   아래 z-score를 기반으로 이상치를 판단함:

   $$
   z_i = \frac{0.6745 \cdot (x_i - \text{Median}(x))}{\text{MAD}}
   $$

   일반적으로 `|z| > 3.5` 이상이면 이상치로 간주

---

### ✅ 장점

* **극단값에 민감하지 않음** (평균 기반 Z-score보다 강인함)
* 비정규분포나 노이즈가 많은 데이터에 효과적

---

### ✅ 예제 코드 (NumPy + Pandas)

```python
import numpy as np
import pandas as pd

def detect_outliers_mad(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    
    if mad == 0:
        return pd.Series([False] * len(series), index=series.index)
    
    modified_z_scores = 0.6745 * (series - median) / mad
    return np.abs(modified_z_scores) > threshold

# 사용 예
outliers = detect_outliers_mad(df['value_col'])
df_cleaned = df[~outliers]
```

---

### 📌 적용 대상

* **센서 데이터**, **재무 시계열**, **환경 모니터링** 등에서 **극단값 필터링**에 적합
* 단일 변수 이상치 탐지에 주로 사용됨

원한다면 다변량 MAD 변형도 설명 가능.

   일반적으로 `|z| > 3.5` 이상이면 이상치로 간주

---

### ✅ 장점

* **극단값에 민감하지 않음** (평균 기반 Z-score보다 강인함)
* 비정규분포나 노이즈가 많은 데이터에 효과적

---

### ✅ 예제 코드 (NumPy + Pandas)

```python
import numpy as np
import pandas as pd

def detect_outliers_mad(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    
    if mad == 0:
        return pd.Series([False] * len(series), index=series.index)
    
    modified_z_scores = 0.6745 * (series - median) / mad
    return np.abs(modified_z_scores) > threshold

# 사용 예
outliers = detect_outliers_mad(df['value_col'])
df_cleaned = df[~outliers]
```

---

### 📌 적용 대상

* **센서 데이터**, **재무 시계열**, **환경 모니터링** 등에서 **극단값 필터링**에 적합
* 단일 변수 이상치 탐지에 주로 사용됨

원한다면 다변량 MAD 변형도 설명 가능.


[돌아가기](../README.md)