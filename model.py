from pandas import DataFrame, concat, read_csv, set_option
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
import numpy as np
import statsmodels.api as sm
from sklearn.pipeline import Pipeline

set_option('display.max_columns', None)


def get_training_set(fname, test_path):
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


def normalize_data(df, norm_cols):
    onhe, scaler = OneHotEncoder(), StandardScaler()
    transforming = make_column_transformer((scaler, norm_cols), remainder="passthrough")
    processed_df = DataFrame(transforming.fit_transform(df))
    result_df = processed_df.iloc[:, [15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]]
    return result_df


def get_model(x_train, y_train):
    model = (sm.OLS(y_train.astype(float), x_train.astype(float))).fit()
    print(model.summary())
    print(model.params)
    return model

def make_predictions(model, x_test, y_test):
    preds = DataFrame(model.predict(x_test.astype(float)))
    combined = concat([x_test, y_test, preds], axis=1, ignore_index=True)
    preds = preds.values.tolist()
    #combined.columns = ['Age', 'Sex', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT/mm3', 'HGB', 'RBC', 'Predicted']
    return combined, preds

if __name__ == '__main__':
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'
    train_df, test_df = get_training_set(train_path, test_path)

    print(list(train_df.columns))
    cleaned_df = clean_data(train_df)
    scale = MinMaxScaler()
    norm_cols = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia', 'MinOfUpperTRange', 'AverageOfUpperTRange',
                 'MaxOfLowerTRange',
                 'MinOfLowerTRange', 'AverageOfLowerTRange', 'RainingDays', 'AverageRainingDays', 'fruitset',
                 'fruitmass', 'seeds']
    onhe_cols = []
    normalized_df = normalize_data(df=cleaned_df, norm_cols=norm_cols)
    x_train = normalized_df.iloc[:, 1:17]
    y_train = normalized_df.iloc[:, 17]
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8)
    model = get_model(x_train,y_train)

    combined, preds = make_predictions(model,x_test,y_test)
    ls = [float(val) for val in y_test]
    r2 = r2_score(ls, preds)
    mse = mean_squared_error(ls, preds)
    mae = mean_absolute_error(ls,preds)

    print(f'r2 score is {r2}')
    print(f'mse is {mse}')
    print(f'mae is {mae}')


    print("Program Execution Complete...")
