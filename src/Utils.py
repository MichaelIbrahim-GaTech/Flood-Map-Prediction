import pandas as pd
import numpy as np
import xarray as xr

def parse_dates(band_str:str)->tuple:
    """handle the case where the band is a datetime64 or a string (e.g. "20030101-20030108") 
    conversion to a datetime64 (some xarry have datetime64 and some have string)

    Args:
        band_str (str): string of the form "20030101-20030108"

    Returns:
        tuple: start_date, end_date
    """
    if isinstance(band_str, np.datetime64):
        start_date = band_str
        end_date = None  
    else:
        start_str, end_str = band_str.split('-')
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
    return start_date, end_date

def batch_predict(models, 
                  X: np.ndarray, 
                  batch_size: int = 10000,
                  is_proba = True) -> np.ndarray:
    """Predicts probabilities in batches to reduce memory usage.

    Parameters
    ----------
    model : Any
        The machine learning model used for prediction. (here random forest)
    X : np.ndarray
        The input features.
    batch_size : int, optional
        The size of each batch for prediction. Default is 10000.

    Returns
    -------
    np.ndarray
        The concatenated predictions from each batch as a numpy array.

    """
    n_samples = X.shape[0]
    y_pred_proba_batches = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X[start:end]
        y_pred_proba_batch = np.zeros(shape=(1,end-start))
        if is_proba:
            for m in models:
                y_pred_proba_batch = y_pred_proba_batch + m.predict_proba(batch)[:, 1]/len(models)
            y_pred_proba_batches.append(y_pred_proba_batch)
        else:
            for m in models:
                y_pred_proba_batch = y_pred_proba_batch + m.predict(batch)/len(models)
            y_pred_proba_batch = y_pred_proba_batch>=(len(models))/2
            y_pred_proba_batches.append(y_pred_proba_batch.astype(int))

    return np.concatenate(y_pred_proba_batches,axis=1).T



def stack_if_exists(X_combined, y_combined, X, y):
    """Stacks the new set of features and targets onto the existing combined arrays.
    
    Parameters
    ----------
    X_combined : np.ndarray or None
        The existing array of combined features, or None if not yet initialized.
    y_combined : np.ndarray or None
        The existing array of combined targets, or None if not yet initialized.
    X : np.ndarray
        The new set of features to add.
    y : np.ndarray
        The new set of targets to add.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The updated combined feature and target arrays.

    """
    if X_combined is None:
        X_combined = X
        y_combined = y
    else:
        X_combined = np.vstack((X_combined, X))
        y_combined = np.hstack((y_combined, y))

    return X_combined, y_combined



def split_time_index(band_index):
    band_index_str = str(band_index)
    return band_index_str.split(":")[0]
