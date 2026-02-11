import pandas as pd

def create_features(df):
    for window in [1, 6, 12]:
        df[f'temp_roll_mean_{window}h'] = (
            df.groupby('machine_id')['temperature']
            .rolling(window).mean().reset_index(0, drop=True)
        )

        df[f'vibration_std_{window}h'] = (
            df.groupby('machine_id')['vibration']
            .rolling(window).std().reset_index(0, drop=True)
        )

    df['temp_lag_1'] = df.groupby('machine_id')['temperature'].shift(1)
    df['vibration_lag_1'] = df.groupby('machine_id')['vibration'].shift(1)

    df = df.dropna()
    return df
