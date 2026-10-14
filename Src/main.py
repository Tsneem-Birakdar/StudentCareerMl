import os
import pandas as pd
from data_loader import DataLoader
from eda import EDA
from preprocessing import DataPreprocessor
from classification import ClassificationSystem

RANDOM_STATE = 42
FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","student_career_success_dataset.csv")
TARGET_COLUMN = "Placement_Status"

class StudentCareerMLSystem:
    def __init__(self,file_path,target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.data_loader = DataLoader(file_path)
        self.df = None
        self.classification = None
        
    def run(self):
        print("STUDENT CAREER SUCCESS ML SYSTEM")
        print("\nLoading Dataset...")
        self.df = self.data_loader.load_data()
        print("\nExploratory Data Analysis...")
        eda = EDA(self.df)
        eda.basic_info(self.df)
        eda.target_distribution(self.target_column)
        eda.numerical_correlation()
        print("\n[3] Preparing Data...")
        preprocessor_manager = DataPreprocessor(self.target_column)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            numeric_features,
            categorical_features
        ) = preprocessor_manager.prepare_data(self.df)
        print("\nTraining samples:",X_train.shape[0])
        print("Testing samples:",X_test.shape[0])
        print("\n Classification...")
        self.classification = ClassificationSystem(preprocessor)
        results = (self.classification.train_and_evaluate(X_train,X_test,y_train,y_test))
        print("\nMODEL COMPARISON")
        print(results.to_string(index=False))
        (
            best_model_name,
            best_model
        ) = self.classification.get_best_model()
        print(f"\nBest Model: {best_model_name}")

if __name__ == "__main__":
    system = StudentCareerMLSystem(file_path=FILE_PATH,target_column=TARGET_COLUMN)
    system.run()