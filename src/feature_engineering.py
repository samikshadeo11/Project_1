import pandas as pd

def create_features(df):
    # IMF advanced features (unique to your dataset)
    for imf in ['IMF_1', 'IMF_2', 'IMF_3']:
        for window in [6, 12]:
            df[f'{imf}_roll_mean_{window}h'] = df.groupby('machine_id')[imf].rolling(window).mean().reset_index(0, drop=True)
            df[f'{imf}_std_{window}h'] = df.groupby('machine_id')[imf].rolling(window).std().reset_index(0, drop=True)

    df['temp_lag_1'] = df.groupby('machine_id')['temperature'].shift(1)
    df['vibration_lag_1'] = df.groupby('machine_id')['vibration'].shift(1)

    df = df.dropna()
    return df
