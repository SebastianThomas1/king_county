# Sebastian Thomas (datascience at sebastianthomas dot de)

# data
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

# metrics
from sklearn.metrics import mean_squared_error

def data_type_info(df):
    '''
    computes a data frame that gives, for each feature, information on its data type, its number of unique
    values, and its number of NA values
    '''

    n_instances = df.shape[0]
    
    n_unique = df.nunique()
    p_unique = n_unique / n_instances
    
    n_na = df.isna().sum()
    p_na = n_na / n_instances
    
    info = pd.concat([df.dtypes, n_unique, p_unique, n_na, p_na], ignore_index=True, axis=1)
    info.index.rename('features')
    info.columns = ['dtype', 'n_unique', 'p_unique', 'n_na', 'p_na']
    
    return info

def cast_data_types(df, int_features=None, float_features=None, bool_features=None, categorical_features=None, ordered_features=None, string_features=None, object_features=None):
    if int_features is not None:
        for feature in int_features:
            df[feature] = df[feature].astype('int')

    if float_features is not None:
        for feature in float_features:
            df[feature] = df[feature].astype('float')

    if bool_features is not None:
        for feature in bool_features:
            df[feature] = df[feature].astype('bool')

    if categorical_features is not None:
        for feature in categorical_features:
            if df[feature].dtype == 'float':
                categories = [category for category in np.sort(df[feature].unique())
                              if not pd.isna(category)]
                # if appropriate, cast categories to integers
                if all([category.is_integer() for category in categories]):
                    categories = [int(category) for category in categories]
            elif df[feature].dtype == 'int':
                categories = np.sort(df[feature].unique())
            elif df[feature].dtype == 'object':
                categories = [category for category in df[feature].unique() if not pd.isna(category)]
            elif df[feature].dtype == 'category':
                categories = df[feature].dtype.categories
            df[feature] = df[feature].astype(CategoricalDtype(categories=categories, ordered=False))

    if ordered_features is not None:
        for feature in ordered_features:
            if df[feature].dtype == 'float':
                categories = [category for category in np.sort(df[feature].unique()) if not np.isnan(category)]
                # if appropriate, cast categories to integers
                if all([category.is_integer() for category in categories]):
                    categories = [int(category) for category in categories]
            elif df[feature].dtype == 'int':
                categories = np.sort(df[feature].unique())
            elif df[feature].dtype == 'category':
                categories = np.sort(df[feature].dtype.categories)
            df[feature] = df[feature].astype(CategoricalDtype(categories=categories, ordered=True))

    if string_features is not None:
        for feature in string_features:
            df[feature] = df[feature].astype('string')
    
    if object_features is not None:
        for feature in object_features:
            df[feature] = df[feature].astype('object')
            
    return df
            
def cast_datetime(df, datetime_features=None, unit=None):
    if datetime_features is not None:
        for feature in datetime_features:
            df[feature] = pd.to_datetime(df[feature], unit=unit)
    
    return df
            
def non_unique_features_of_duplicates(df):
    duplicates = df[df.index.duplicated(keep=False)]
    number_of_values = duplicates.groupby(duplicates.index).nunique()
    number_of_non_unique_values = (number_of_values > 1).sum()
    return np.array(number_of_non_unique_values[number_of_non_unique_values != 0].index)

def logarithmize_features(df, to_be_logarithmized, log1p=True):
    for entry in to_be_logarithmized:
        (feature, logarithmized_feature) = entry if type(entry) == tuple else (entry, entry + ' log')
        df[logarithmized_feature] = np.log1p(df[feature]) if log1p else np.log(df[feature])
    
    return df

def bin_features(df, bin_data):
    for (key, (bins, labels)) in bin_data.items():
        (feature, binned_feature) = key if type(key) == tuple else (key, key + ' bin')
        df[binned_feature] = pd.cut(df[feature], bins, labels=labels)
    
    return df

def coarsen_categories(df, coarsen_data):
    for (key, coarsen_map) in coarsen_data.items():
        (feature, coarsened_feature) = key if type(key) == tuple else (key, key + ' cat')
        df[coarsened_feature] = df[feature].map(coarsen_map).astype('category')
        
    return df

def reindex_columns(df, columns):
    return df.reindex(columns=columns)

def lower_quartile(a):
    return np.percentile(a, 25)

def upper_quartile(a):
    return np.percentile(a, 75)

def whiskers(a):
    lq = lower_quartile(a)
    uq = upper_quartile(a)
    ir = uq - lq
    
    lower_whisker = np.min(a[a >= lq - 1.5*ir])
    upper_whisker = np.max(a[a <= uq + 1.5*ir])
    
    return (lower_whisker, upper_whisker)

def boxplot_statistics(a):
    (lower_whisker, upper_whisker) = whiskers(a) 

    return (lower_whisker, lower_quartile(a), np.median(a), upper_quartile(a), upper_whisker)

def root_mean_squared_error(y, y_pred):    
    '''returns root mean squared error'''
    return np.sqrt(mean_squared_error(y, y_pred))

def median_absolute_error(y, y_pred):    
    '''returns median absolute error'''
    return np.median(np.abs(y - y_pred))

def mean_percentage_error(y, y_pred):
    '''returns mean percentage error'''
    return np.mean((y - y_pred) / y)

def mean_absolute_percentage_error(y, y_pred):    
    '''returns mean percentage error'''
    return np.mean(np.abs((y - y_pred) / y))

#class IntTyper(BaseEstimator, TransformerMixin):
#
#    def __init__(self, features):
#        self.features = features
#    
#    def fit(self, X, y=None):
#        return self
#    
#    def transform(self, X, y=None):
#        if self.features is not None:
#            for feature in self.features:
#                X[feature] = X[feature].astype('int')
#        return X
#
#class FloatTyper(BaseEstimator, TransformerMixin):
#
#    def __init__(self, features):
#        self.features = features
#    
#    def fit(self, X, y=None):
#        return self
#    
#    def transform(self, X, y=None):
#        if self.features is not None:
#            for feature in self.features:
#                X[feature] = X[feature].astype('float')
#        return X
#
#class BoolTyper(BaseEstimator, TransformerMixin):
#    
#    def __init__(self, features):
#        self.features = features
#    
#    def fit(self, X, y=None):
#        return self
#    
#    def transform(self, X, y=None):
#        if self.features is not None:
#            for feature in self.features:
#                X[feature] = X[feature].astype('bool')
#        return X
#    
#class CategoricalTyper(BaseEstimator, TransformerMixin):
#    
#    def __init__(self, features):
#        self.features = features
#    
#    def fit(self, X, y=None):
#        return self
#    
#    def transform(self, X, y=None):
#        if self.features is not None:
#            for feature in self.features:
#                categories = [category for category in np.sort(X[feature].unique()) if not pd.isna(category)]
#                # if appropriate, cast categories to integers
#                if all([category.is_integer() for category in categories]):
#                    categories = [int(category) for category in categories]
#                X[feature] = X[feature].astype(CategoricalDtype(categories=categories, ordered=False))
#                #X[feature] = X[feature].astype('category')
#        return X
#    
#class OrderedTyper(BaseEstimator, TransformerMixin):
#    
#    def __init__(self, features):
#        self.features = features
#    
#    def fit(self, X, y=None):
#        return self
#    
#    def transform(self, X, y=None):
#        if self.features is not None:
#            for feature in self.features:
#                # currently only works for integers?
#                categories = [category for category in np.sort(X[feature].unique()) if not pd.isna(category)]
#                # if appropriate, cast categories to integers
#                if all([category.is_integer() for category in categories]):
#                    categories = [int(category) for category in categories]
#                X[feature] = X[feature].astype(CategoricalDtype(categories=np.sort(X[feature].unique()),
#                                                                ordered=True))
#        return X
#
#class ObjectTyper(BaseEstimator, TransformerMixin):
#    
#    def __init__(self, features):
#        self.features = features
#    
#    def fit(self, X, y=None):
#        return self
#    
#    def transform(self, X, y=None):
#        if self.features is not None:
#            for feature in self.features:
#                X[feature] = X[feature].astype('object')
#        return X

