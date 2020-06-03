# Sebastian Thomas (datascience at sebastianthomas dot de)

# data
import numpy as np


def clean_king_county(df):
    # NA values of feature 'sqft_basement'
    df['sqft_basement'].fillna(df['sqft_living'] - df['sqft_above'], inplace=True)

    # NA values and wrong values of feature 'yr_renovated'
    df['yr_renovated'].replace({0.: np.nan}, inplace=True)
    df['yr_renovated'].fillna(df['yr_built'], inplace=True)
    
    # NA values of feature 'waterfront', 'view'
    for feature in ['waterfront', 'view']:
        df[feature].fillna(0, inplace=True)
    
    # wrong value of feature 'bedrooms'
    if 2402100895 in df.index:
        df.loc[2402100895, 'bedrooms'] = 3
        
    return df