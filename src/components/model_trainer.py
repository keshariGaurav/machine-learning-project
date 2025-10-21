import os
import sys
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.utils import  evaluate_model, save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join("artifacts", "model.pkl")
  
class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
    
  def initiate_model_trainer(self, train_array, test_array):
    try:
      logging.info("Splitting Training and Test input Data")
      x_train, y_train, x_test, y_test = (
        train_array[:, :-1],
        train_array[:, -1],
        test_array[:, :-1],
        test_array[:, -1],
      )
      models = {
          "Linear Regression": LinearRegression(),
          "K-Neighbors Regressor": KNeighborsRegressor(),
          "Decision Tree": DecisionTreeRegressor(),
          "Random Forest Regressor": RandomForestRegressor(),
          "XGBRegressor": XGBRegressor(), 
          "CatBoosting Regressor": CatBoostRegressor(verbose=False),
          "AdaBoost Regressor": AdaBoostRegressor(),
          "Gradient Boosting": GradientBoostingRegressor()
      }
      
      params = {
          "Linear Regression": {
              "fit_intercept": [True, False],
              "positive": [True, False]
          },
          
          "K-Neighbors Regressor": {
              "n_neighbors": [3, 5, 7, 9, 11],
              "weights": ["uniform", "distance"],
              "p": [1, 2]  # 1 for Manhattan, 2 for Euclidean
          },
          
          "Decision Tree": {
              "criterion": ["squared_error", "friedman_mse", "absolute_error"],
              # "max_depth": [None, 5, 10, 20, 30],
              # "min_samples_split": [2, 5, 10],
              # "min_samples_leaf": [1, 2, 4],
              "max_features": [None, "sqrt", "log2"]
          },
          
          "Random Forest Regressor": {
              "n_estimators": [100, 200, 300],
              "criterion": ["squared_error", "friedman_mse", "absolute_error"],
              # "max_depth": [None, 10, 20, 30],
              # "min_samples_split": [2, 5, 10],
              # "min_samples_leaf": [1, 2, 4],
              # "max_features": ["sqrt", "log2", None],
              "bootstrap": [True, False]
          },
          
          "XGBRegressor": {
              "n_estimators": [100, 200, 300],
              # "learning_rate": [0.01, 0.05, 0.1, 0.2],
              # "max_depth": [3, 5, 7, 10],
              # "subsample": [0.6, 0.8, 1.0],
              # "colsample_bytree": [0.6, 0.8, 1.0],
              # "gamma": [0, 0.1, 0.2, 0.3],
              "reg_alpha": [0, 0.01, 0.1],
              "reg_lambda": [0.5, 1, 2]
          },
          
          "CatBoosting Regressor": {
              "iterations": [200, 500, 800],
              # "depth": [4, 6, 8, 10],
              # "learning_rate": [0.01, 0.05, 0.1],
              # "l2_leaf_reg": [1, 3, 5, 7],
              # "border_count": [32, 64, 128]
          },
          
          "AdaBoost Regressor": {
              "n_estimators": [50, 100, 200],
              # "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
              "loss": ["linear", "square", "exponential"]
          },
          
          "Gradient Boosting": {
              "n_estimators": [100, 200, 300],
              # "learning_rate": [0.01, 0.05, 0.1],
              # "max_depth": [3, 5, 7],
              # "min_samples_split": [2, 5, 10],
              # "min_samples_leaf": [1, 2, 4],
              # "subsample": [0.6, 0.8, 1.0],
              "max_features": ["sqrt", "log2", None]
          }
      }

      
      model_report: dict=evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, param=params)
      best_model_score = max(sorted(model_report.values()))
      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]
      best_model = models[best_model_name]
      if best_model_score < 0.6:
        raise CustomException("No Best Model Found")
      
      logging.info("Best model found on both training and testing dataset.")
      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )
      predicted=best_model.predict(x_test)
      r2_score_val = r2_score(y_test, predicted)
      return r2_score_val
      
    except Exception as e:
      raise CustomException(e, sys)
