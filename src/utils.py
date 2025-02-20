import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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

        outliers = df_col[(df_col < low_lim) | (df_col > up_lim)].index.tolist()
        outliers_dict[col] = outliers

        print(f'Column: {col}   |   Outliers: {len(outliers)}')

    return outliers_dict
    

def lof_outlier_detection(df, cols, n_neighbors=20, contamination=0.05):
    """
    Detects outliers in financial columns using LOF

    Returns:
    list: A list of indices of where the outliers were detected.
    """
    scaler = StandardScaler()
    df_cols = scaler.fit_transform(df[cols])

    #df_cols = df[cols].values

    # Create the model and fit it
    lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outliers = lof_model.fit_predict(df_cols)

    # Get the indices of the outliers (-1 indicates an outlier)
    outlier_indices = np.where(outliers == -1)[0].tolist()

    print(f'Total outliers detected: {len(outlier_indices)}')
    return outlier_indices


def iso_forest_outlier_detection(df, cols, contaimination=0.05):
    """
    Detects outliers in financial columns using Isolation Forest

    Returns:
    dict: A dictionary with the financial column names as keys and the number of outliers as values.
    """
    scaler = StandardScaler()
    df_cols = scaler.fit_transform(df[cols])

    #df_cols = df[cols].values

    # Create the model and fit it
    iso_forest_model = IsolationForest(contamination=contaimination, random_state=42)
    outliers = iso_forest_model.fit_predict(df_cols)

    # Get the indices of the outliers (-1 indicates an outlier)
    outlier_indices = np.where(outliers == -1)[0].tolist()

    print(f'Total Outliers: {len(outlier_indices)}')
    return outlier_indices
