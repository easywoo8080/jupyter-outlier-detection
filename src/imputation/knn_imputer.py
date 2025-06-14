from typing import Literal
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def knn_impute(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    weights: Literal["uniform", "distance"] = "uniform",
) -> pd.DataFrame:
    """
    KNN 기반 결측치 보간 함수

    Parameters
    ----------
    df : pd.DataFrame
        결측치를 포함한 데이터프레임
    n_neighbors : int
        이웃 수 (기본값: 5)
    weights : str
        'uniform' 또는 'distance'

    Returns
    -------
    pd.DataFrame
        보간 완료된 데이터프레임
    """
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)
