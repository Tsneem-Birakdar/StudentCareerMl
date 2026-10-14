import pandas as pd
import numpy as nb
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import DataPreprocessor

class EDA:
    def __init__(self, df,output_path="Output/plots"):
        self.df = df
        self.output_path = output_path
        sns.set_theme(style="whitegrid")
    
    def basic_info(self, df):
        print("\n DATASET INFO")
        print("Shape:", df.shape)
        print("Columns:",df.columns.tolist())
        print("Data Types:",df.dtypes)
        print("information:",df.info())
        print("Missing Values:",df.isnull().sum())
        print("Duplicate Rows:",df.duplicated().sum())
        print("describe the data",df.describe())

    def unique(self,df):
        for column in df.columns:
            print(df[column].unique())

    def target_distribution(self, target_column):
        print("\nTARGET DISTRIBUTION ")
        print(self.df[target_column].value_counts())
        plt.figure(figsize=(7, 5))
        sns.countplot(data=self.df, x=target_column)
        plt.title("Placement Status Distribution")
        plt.xlabel("Placement Status")
        plt.ylabel("Number of Students")
        plt.tight_layout()
        plt.savefig("plots/target_distribution.png")
        plt.close()

    def numerical_correlation(self):
        numeric_df = self.df.select_dtypes(include="number")
        plt.figure(figsize=(14, 10))
        sns.heatmap(numeric_df.corr(),cmap="coolwarm",annot=True)
        plt.title("Numerical Features Correlation")
        plt.tight_layout()
        plt.savefig("plots/numerical_correlation.png")

    

    

    
        

        