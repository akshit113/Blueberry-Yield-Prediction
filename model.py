from pandas import DataFrame, read_csv, set_option
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
import numpy as np
from sklearn.pipeline import Pipeline

set_option('display.max_columns', None)


def get_training_set(fname,test_path):
    train_df, test_df = read_csv(fname),read_csv(test_path)
    # print(train_df.head(30).to_string())
    return train_df, test_df


def clean_data(df):
    df_has_nans = df.isna().values.any()
    if df_has_nans:
        imputer = SimpleImputer(missing_values='NaN', strategy='mean', axis=0)
        imputer.fit(df)
        imputer.transform(df)
    return df


def normalize_data(df, onhe_cols, norm_cols):
    onhe, scaler = OneHotEncoder(), StandardScaler()
    transforming = make_column_transformer((onhe, onhe_cols), (scaler, norm_cols), remainder="passthrough")
    processed_df = DataFrame(transforming.fit_transform(df))
    return processed_df


if __name__ == '__main__':
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'
    train_df,test_df = get_training_set(train_path,test_path)


    print(list(train_df.columns))
    cleaned_df = clean_data(train_df)
    scale = MinMaxScaler()
    norm_cols = ['clonesize','honeybee','bumbles','andrena', 'osmia', 'MinOfUpperTRange', 'AverageOfUpperTRange', 'MaxOfLowerTRange',
                 'MinOfLowerTRange', 'AverageOfLowerTRange', 'RainingDays', 'AverageRainingDays', 'fruitset',
                 'fruitmass', 'seeds']
    onhe_cols = []
    normalized_df = normalize_data(df=cleaned_df, onhe_cols=onhe_cols,norm_cols=norm_cols)
    x_train = normalized_df.iloc[:,[1,17]]
    y_train = normalized_df.iloc[:,:-1]
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size= 0.2, train_size=0.8)


    print("Program Execution Complete...")
