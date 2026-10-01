import numpy as np
import pandas as pd
LEAKAGE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
TARGET = 'Machine failure'
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
    df['wear_x_torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
    return df
