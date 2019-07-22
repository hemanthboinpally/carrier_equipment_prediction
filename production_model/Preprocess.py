import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Preprocess:

    def __init__(self, data, label_encoder_dict, standardization_dict, dict_vectorizer, mode_dict, nulls_dict, categorical_cols, noncategorical_cols):
        self.df = data
        self.label_encoder_dict = label_encoder_dict
        self.standardization_dict = standardization_dict
        self.dict_vectorizer = dict_vectorizer
        self.mode_dict = mode_dict
        self.nulls_dict = nulls_dict
        self.categorical_cols = categorical_cols
        self.noncategorical_cols = noncategorical_cols


    def get_boolean_flags(self, cols, df):
        set_of_flags = set()
        for col in cols:
            set_of_flags.update(list(df[col].unique()))

        return set_of_flags

    def clean_data(self, cols, df, isboolean=True):
        if isboolean:
            set_of_flags = self.get_boolean_flags(cols, df)
            for col in cols:
                df[col] = np.where(df[col].isin(set_of_flags), True, False)

    def log_transformation(self,cols, df,label='_log',replace_cols=True,replace=True):

        log_df = df[cols].copy()
        for col in cols:
            to_replace_mask = df.index[(df[col] <= 0) | df[col].isnull()]
            replace_val = df.loc[~df.index.isin(to_replace_mask), [col]].median()
            print('Inserting %s into col %s' % (replace_val[0], col))
            log_df.loc[to_replace_mask, [col]] = replace_val[0]
            log_df[col] = np.log(log_df[col])
            log_df.rename(columns={col: col + label}, inplace=True)
        df = pd.concat([df, log_df], axis=1)

        if replace_cols:
            df = self.drop_columns(df,label,cols=cols,replace=True)
        elif replace:
            self.df = df.copy()
        else:
            return df.copy()

    def drop_columns(self,df,label,cols,replace=False,):

        for col in cols:
            if col + label in df.columns:
                df.drop(col + label, axis=1, inplace=True)
        if replace:
            self.df = df
        else:
            return df

    def set_categorical_and_non_cat(self):

        ###### Categorical vs. Continuous Column Assignments
        # Create a boolean mask for categorical columns
        categorical_feature_mask = self.df.dtypes == object

        # Get list of categorical column names
        categorical_cols = self.df.columns[categorical_feature_mask].tolist()

        # Get list of non-categorical column names
        noncategorical_cols = self.df.columns[~categorical_feature_mask].tolist()

        self.categorical_cols = categorical_cols
        self.noncategorical_cols = noncategorical_cols


    def label_encode(self,data,cols,label,replace=False,label_encoder=None,le_dict=None,replace_cols=False):

        if le_dict is None:
            le_dict = {}

        if label_encoder is None:
            le = LabelEncoder()

        else:
            le = label_encoder

        for col in cols:
            ## casting to string because we have nans

            if replace_cols:
                data[col] = le.fit_transform(data[col].astype(str))
            else:
                data[col + label] = le.fit_transform(data[col].astype(str))
        if replace:
            self.label_encoder_dict = le_dict.copy()
        else:
            return le_dict








