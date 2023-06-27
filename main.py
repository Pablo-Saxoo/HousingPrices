import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import pathlib


def load(train_path, test_path):
    """Load dataset"""

    try:
        train_data = pd.read_csv(train_path, index_col="Id")
        test_data = pd.read_csv(test_path, index_col="Id")
    except FileNotFoundError:
        return f"File not found"

    return train_data, test_data


def pip_prep(train_data):
    """Preprocessing data, model checking, return mean absolute error"""

    y = train_data.SalePrice
    train_data.drop(
        columns=["SalePrice", "RoofMatl", "Condition2", "Functional"],
        inplace=True,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        train_data, y, test_size=0.3, random_state=42
    )

    categorical_col = [col for col in X_train.columns if X_train[col].dtype == "object"]
    num_col = [
        col for col in X_train.columns if X_train[col].dtype in ["int64", "float64"]
    ]
    my_cols = num_col + categorical_col

    X_train = X_train[my_cols]
    X_valid = X_valid[my_cols]


    numeric_prep = Pipeline(
        steps=[
            ("mean_imp", SimpleImputer(strategy="median")),
        ]
    )

    cat_prep = Pipeline(
        steps=[
            ("freq_val", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_prep, num_col),
            ("categorical", cat_prep, categorical_col),
        ]
    )

    return X_train, X_valid, y_train, y_valid, preprocessor


def make_pred(X_train, X_valid, y_train, y_valid, preprocessor, model):

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_valid)
    mae = mean_absolute_error(predictions, y_valid)

    return mae


def mean_diff_plt(mod_val, models):

    plt.figure()
    plt.title("Models with MAE")
    plt.xlabel("Mean absolute error")
    sns.barplot(x=mod_val, y=models)
    plt.show()



if __name__ == "__main__":

    TRAIN_PATH = pathlib.Path("train.csv")
    TEST_PATH = pathlib.Path("test.csv")

    train_df, test_df = load(TRAIN_PATH, TEST_PATH)

    X_train, X_valid, y_train, y_valid, preprocessor = pip_prep(train_df)


    # # Selecting the best model:
    model = RandomForestRegressor(n_estimators=500, max_features="auto", max_depth=8)
    randomf_score = make_pred(X_train, X_valid, y_train, y_valid, preprocessor, model)

    model = SVC()
    svc_score = make_pred(X_train, X_valid, y_train, y_valid, preprocessor, model)

    model = KNeighborsRegressor(n_neighbors=20, weights="distance", algorithm="kd_tree")
    kn_score = make_pred(X_train, X_valid, y_train, y_valid, preprocessor, model)

    model = AdaBoostRegressor(random_state=0, n_estimators=40)
    ada_score = make_pred(X_train, X_valid, y_train, y_valid, preprocessor, model)

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=1,
        random_state=0,
        loss="squared_error",
    )
    gradb_score = make_pred(X_train, X_valid, y_train, y_valid, preprocessor, model)

    clf = xgb.XGBRegressor(n_estimators=400, learning_rate=0.06)
    xgb_score = make_pred(X_train, X_valid, y_train, y_valid, preprocessor, clf)


    models = [
        "XGBRegressor",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "KNeighborsRegressor",
        "AdaBoostRegressor",
        "SVC",
    ]

    mod_val = [
        xgb_score,
        randomf_score,
        gradb_score,
        kn_score,
        ada_score,
        svc_score,
    ]

    # Plot MAE for different models
    mean_diff_plt(mod_val, models)