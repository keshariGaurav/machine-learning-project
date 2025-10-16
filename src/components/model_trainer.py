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
      
      model_report: dict=evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)
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
