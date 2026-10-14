import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:

    def __init__(self, target_column):
        self.target_column = target_column
        self.numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())]
        )
        self.categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))]
        )

    def prepare_data(self, df):
        excluded_columns = ["Student_ID","Company_Tier","Career_Field","Placement_Mode","Starting_Salary_USD"]
        X = df.drop( columns=[self.target_column] + excluded_columns,errors="ignore")
        y = df[self.target_column]
        numeric_features = X.select_dtypes( include=["int64", "float64"] ).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    self.numeric_transformer,
                    numeric_features
                ),
                (
                    "categorical",
                    self.categorical_transformer,
                    categorical_features
                )
            ]
        )
        X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.20,random_state=42)
        return (X_train,X_test,y_train,y_test,preprocessor,numeric_features,categorical_features)