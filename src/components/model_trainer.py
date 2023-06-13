import os
import sys

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from src.utils import save_object, evaluate_models

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.data_transformation import DataTransformationConfig, DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")

            Xtrain, ytrain, Xtest, ytest = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(Xtrain= Xtrain, ytrain = ytrain, Xtest = Xtest, ytest= ytest, models = models)

            #getting the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #getting the best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Besr found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file,
                obj = best_model
            )

            predicted = best_model.predict(Xtest)

            r2_square = r2_score(ytest, predicted)

            result = f"The best models is {best_model_name} with performance of {r2_square*100}%"

            return result

        except Exception as e:
            raise CustomException
        


if __name__ =="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_tranformation = DataTransformation()
    train_arr, test_arr,_ = data_tranformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
            
