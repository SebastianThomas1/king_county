# data
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


def engineer_king_county(df):
    df['month'] = df['date'].dt.strftime('%Y-%m')
    df['month'] = df['month'].astype(CategoricalDtype(categories=np.sort(df['month'].unique()), ordered=True))
    df['room_size'] = df['sqft_living'] / (df['bathrooms'] + df['bedrooms'] + 1)
    df['base_area'] = df['sqft_living'] / df['floors']
    df['has_basement'] = df['sqft_basement'] > 0
    df['bathrooms_ratio'] = df['bathrooms'] / df['bedrooms']
    df['is_renovated'] = df['yr_renovated'] >= 1995

    return df