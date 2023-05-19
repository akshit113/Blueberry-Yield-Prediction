from pandas import DataFrame, concat, read_csv, set_option
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
import numpy as np
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from seaborn import lineplot, regplot

set_option('display.max_columns', None)


def get_data(fname, test_path):
    train_df, test_df = read_csv(fname), read_csv(test_path)
    # print(train_df.head(30).to_string())
    return train_df, test_df


def clean_data(df):
    df_has_nans = df.isna().values.any()
    if df_has_nans:
        imputer = SimpleImputer(missing_values='NaN', strategy='mean', axis=0)
        imputer.fit(df)
        imputer.transform(df)
    return df


def normalize_data(train_df, test_df, norm_cols):
    onhe, scaler = OneHotEncoder(), StandardScaler()
    transforming = make_column_transformer((scaler, norm_cols), remainder="passthrough")
    processed_train_df = DataFrame(transforming.fit_transform(train_df))
    result_train_df = processed_train_df.iloc[:, [15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]]
    processed_test_df = DataFrame(transforming.fit_transform(test_df))
    result_test_df = processed_test_df.iloc[:, [15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]]
    return result_train_df, result_test_df


def get_model(x_train, y_train):
    model = (sm.OLS(y_train.astype(float), x_train.astype(float))).fit()
    print(model.summary())
    print(model.params)
    return model


def make_predictions(model, x_test, y_test=None):
    preds = DataFrame(model.predict(x_test.astype(float)))
    combined = []
    if y_test is not None:
        combined = concat([x_test, y_test, preds], axis=1, ignore_index=True)
    preds = DataFrame(list(preds.values))
    return combined, preds


if __name__ == '__main__':
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'
    train_df, test_df = get_data(train_path, test_path)

    print(list(train_df.columns))
    cleaned_train_df = clean_data(train_df)
    cleaned_test_df = clean_data(test_df)

    scale = MinMaxScaler()
    norm_cols = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia', 'MinOfUpperTRange', 'AverageOfUpperTRange',
                 'MaxOfLowerTRange',
                 'MinOfLowerTRange', 'AverageOfLowerTRange', 'RainingDays', 'AverageRainingDays', 'fruitset',
                 'fruitmass', 'seeds']
    onhe_cols = []
    result_train_df, result_test_df = normalize_data(train_df=cleaned_train_df, test_df=cleaned_test_df,
                                                     norm_cols=norm_cols)
    x_train = result_train_df.iloc[:, 1:17]
    y_train = result_train_df.iloc[:, 17]
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8)
    model = get_model(x_train, y_train)

    combined, preds = make_predictions(model, x_test, y_test)
    ls = DataFrame([float(val) for val in y_test])
    final_df = concat([ls, preds], axis=1)
    final_df.columns = ['true', 'predicted']

    r2 = r2_score(ls, preds)
    mse = mean_squared_error(ls, preds)
    mae = mean_absolute_error(ls, preds)

    regplot(x='true', y='predicted', data=final_df)
    import matplotlib.pyplot as plt

    plt.show()

    print(f'r2 score is {r2}')
    print(f'mse is {mse}')
    print(f'mae is {mae}')

    #### Create Submissions
    X = result_test_df.iloc[:, 1:]
    combined, preds = make_predictions(model, X)
    final_df = concat([test_df['id'], preds], axis=1)
    final_df.columns = ['id', 'yield']
    fpath = f'MAE '+str(round(mae,2))+'.csv'
    final_df.to_csv(path_or_buf=fpath, index=False)

    print("Program Execution Complete...")
