import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def iqr_outlier_detection(df, cols):
    """
    Detects outliers in financial columns using IQR

    Returns:
    dict: A dictionary with the financial column names as keys and the number of outliers as values.
    """
    outliers_dict = {}

    for col in cols:
        df_col = df[col]
        # Calculating Q1, Q2, Q3 and IQR
        Q1 = np.percentile(df_col, 25, method = 'midpoint') 
        Q2 = np.percentile(df_col, 50, method = 'midpoint') 
        Q3 = np.percentile(df_col, 75, method = 'midpoint') 

        IQR = Q3 - Q1 

        # Find the lower and upper limits.
        low_lim = Q1 - 1.5 * IQR
        up_lim = Q3 + 1.5 * IQR

        outliers =[]
        for x in df_col:
            if ((x> up_lim) or (x<low_lim)):
                    outliers.append(float(x))
        outliers_dict[col] = outliers

        print(f'Column: {col}   |   Outliers: {len(outliers)}')
    return outliers_dict


def lof_outlier_detection(df, cols, n_neighbors=50, contaimination=0.05):
    """
    Detects outliers in financial columns using LOF

    Returns:
    dict: A dictionary with the financial column names as keys and the number of outliers as values.
    """
    outliers_dict = {}

    for col in cols:
        df_col = df[col].values.reshape(-1, 1)

        # Create the model and fit it
        lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contaimination)
        outliers = lof_model.fit_predict(df_col)

        # Get the indices of the outliers (-1 indicates an outlier)
        outlier_indices = np.where(outliers == -1)[0].tolist()
        outliers_dict[col] = outlier_indices

        print(f'Column: {col}   |   Outliers: {len(outliers)}')
    return outliers_dict


def iso_forest_outlier_detection(df, cols, contaimination=0.05):
    """
    Detects outliers in financial columns using Isolation Forest

    Returns:
    dict: A dictionary with the financial column names as keys and the number of outliers as values.
    """
    outliers_dict = {}

    for col in cols:
        df_col = df[col].values.reshape(-1, 1)

        # Create the model and fit it
        iso_forest_model = IsolationForest(contamination=contaimination, random_state=42)
        outliers = iso_forest_model.fit_predict(df_col)

        # Get the indices of the outliers (-1 indicates an outlier)
        outlier_indices = np.where(outliers == -1)[0].tolist()
        outliers_dict[col] = outlier_indices

        print(f'Column: {col}   |   Outliers: {len(outliers)}')
    return outliers_dict
