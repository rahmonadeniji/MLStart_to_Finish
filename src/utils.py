import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score


from src.exception import CustomException

#saving objects(model, transformaion) as pkl files
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)

#evaluating models function
def evaluate_models(Xtrain, ytrain, Xtest, ytest, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(Xtrain, ytrain)

            ytrain_pred = model.predict(Xtrain)

            ytest_pred = model.preict(Xtest)

            train_model_score = r2_score(y_train, ytrain_pred)

            test_model_score = r2_score(ytest, ytest_pred)

            report[list(model.keys())[i]] = test_model_score
        
        return report

    except Exception as e:
        raise CustomException(e,sys)



#loading model function
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)



    