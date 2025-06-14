"""
XGBoost 입력 데이터 요건
------------------------

1. 입력 데이터 (X)
   - 수치형 또는 One-hot 인코딩된 범주형
   - 2차원 배열 (샘플 수 × 특성 수)
   - 예시: [[5.1, 3.5, 1.4], [4.9, 3.0, 1.4], ...]

2. 목표값 (y)
   - 1차원 벡터
     - 회귀: 연속값 (예: 집값, 온도 등)
     - 분류: 클래스 라벨 (예: 0/1 또는 0/1/2 등)

3. 결측치(NaN)
   - XGBoost는 NaN 처리가 가능하지만, 사전 파악 및 정제 권장

4. 특성 스케일링
   - 불필요 (결정트리 기반)

전처리 주의사항
---------------
- 범주형 데이터는 pd.get_dummies() 또는 LabelEncoder로 변환
- 문자열/날짜형은 수치형으로 변환
- 고차원 데이터는 차원 축소 또는 중요도 기반 선택 권장
"""

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


def train_xgboost_classifier(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = 3,
    eta: float = 0.1,
    num_round: int = 50,
) -> float:
    """
    XGBoost 다중 클래스 분류 학습 및 평가 함수
    단일 데이터
    Parameters
    ----------
    X : np.ndarray
        입력 특성 데이터 (2차원 배열)
    y : np.ndarray
        정답 라벨 (1차원 배열, 정수형 클래스 라벨)
    test_size : float
        테스트 데이터 비율 (0~1)
    random_state : int
        데이터 분할 시드 고정용
    max_depth : int
        트리 최대 깊이
    eta : float
        학습률
    num_round : int
        부스팅 반복 횟수

    Returns
    -------
    float
        테스트 데이터 정확도 (accuracy)
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # DMatrix 생성
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 하이퍼파라미터 설정
    params = {
        "objective": "multi:softmax",  # 다중 클래스 분류
        "num_class": len(np.unique(y)),
        "eval_metric": "merror",  # 오류율 평가
        "max_depth": max_depth,
        "eta": eta,
        "seed": random_state,
    }

    # 모델 학습
    model = xgb.train(params, dtrain, num_round)

    # 예측 및 평가
    y_pred = model.predict(dtest)
    accuracy = accuracy_score(y_test, y_pred)

    return float(accuracy)


def train_xgboost_regressor(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = 3,
    eta: float = 0.1,
    num_round: int = 50,
) -> float:
    """
    XGBoost 회귀 학습 및 평가 함수
    연속 데이터
    Parameters
    ----------
    X : np.ndarray
        입력 특성 데이터 (2차원 배열)
    y : np.ndarray
        연속형 목표값 (1차원 배열)
    test_size : float
        테스트 데이터 비율 (0~1)
    random_state : int
        데이터 분할 시드 고정용
    max_depth : int
        트리 최대 깊이
    eta : float
        학습률
    num_round : int
        부스팅 반복 횟수

    Returns
    -------
    float
        테스트 데이터 RMSE (Root Mean Squared Error)
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # DMatrix 생성
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 하이퍼파라미터 설정
    params = {
        "objective": "reg:squarederror",  # 회귀용 손실함수
        "max_depth": max_depth,
        "eta": eta,
        "seed": random_state,
    }

    # 모델 학습
    model = xgb.train(params, dtrain, num_round)

    # 예측 및 평가
    y_pred = model.predict(dtest)
    mse = mean_squared_error(y_test, y_pred)  # squared=True 기본값 (MSE)
    rmse = np.sqrt(mse)

    return rmse
