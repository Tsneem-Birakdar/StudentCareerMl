import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer

class ClassificationSystem:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }
        self.param_grids = {

            "Logistic Regression": {
              "model__C": [0.01, 0.1, 1, 10],
              "model__solver": ["lbfgs", "liblinear"]
            },
           "Decision Tree": {
              "model__max_depth": [None, 5, 10, 20],
              "model__min_samples_split": [2, 5, 10],
              "model__min_samples_leaf": [1, 2, 4],
              "model__criterion": ["gini", "entropy"]
           },
          "Random Forest": {
              "model__n_estimators": [100, 200],
              "model__max_depth": [None, 10, 20],
              "model__min_samples_split": [2, 5],
              "model__min_samples_leaf": [1, 2],
              "model__max_features": ["sqrt", "log2"]
            }
        }
        self.trained_models = {}
        self.results = []
        
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        positive_class = "Placed"
        f1_scorer = make_scorer(f1_score, pos_label=positive_class)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, model in self.models.items():
            print(f"\nTuning {model_name}")

            pipeline = Pipeline(
                 steps=[
                      ("preprocessing", clone(self.preprocessor)),
                      ("model", model)
                    ]
                )

            grid_search = GridSearchCV(
               estimator=pipeline,
               param_grid=self.param_grids[model_name],
               cv=cv,
               scoring=f1_scorer,
               n_jobs=-1,
               refit=True
            )

            grid_search.fit(X_train, y_train)

            best_pipeline = grid_search.best_estimator_

            print("\nBest Parameters:")
            print(grid_search.best_params_)

            print("\nBest CV F1-Score:")
            print(grid_search.best_score_)

            y_pred = best_pipeline.predict(X_test)
            probabilities = best_pipeline.predict_proba(X_test)

            class_index = list(best_pipeline.classes_).index(positive_class)
            y_probability = probabilities[:, class_index]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, pos_label=positive_class, zero_division=0
            )
            recall = recall_score(
               y_test, y_pred, pos_label=positive_class, zero_division=0
            )
            f1 = f1_score(
              y_test, y_pred, pos_label=positive_class, zero_division=0
            )

            y_test_binary = (y_test == positive_class).astype(int)
            auc = roc_auc_score(y_test_binary, y_probability)

            self.results.append({
                 "Model": model_name,
                 "CV F1-Score": grid_search.best_score_,
                 "Accuracy": accuracy,
                 "Precision": precision,
                 "Recall": recall,
                 "F1-Score": f1,
                 "ROC-AUC": auc
            })

            self.trained_models[model_name] = best_pipeline

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))

            cm = confusion_matrix(
                y_test, y_pred, labels=["Not Placed", "Placed"]
            )

            plt.figure(figsize=(6, 5))
            sns.heatmap(
               cm,
               annot=True,
               fmt="d",
               cmap="Blues",
               xticklabels=["Not Placed", "Placed"],
               yticklabels=["Not Placed", "Placed"]
            )
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.show()
            plt.close()

        results_df = pd.DataFrame(self.results)
        return results_df

    
    def get_best_model(self):
        results_df = pd.DataFrame(self.results)
        best_index = results_df["F1-Score"].idxmax()
        best_model_name = results_df.loc[best_index,"Model"]
        return (best_model_name,self.trained_models[best_model_name])