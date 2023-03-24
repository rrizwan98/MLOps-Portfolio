import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred =  model.predict(X_test)

            ## Train predicted score
            train_cm=confusion_matrix(y_train,y_train_pred)
            train_model_acc_score = accuracy_score(y_train,y_train_pred)
            train_model_f1_score = f1_score(y_train,y_train_pred)

            ## Test predicted score
            test_cm=confusion_matrix(y_test,y_test_pred)
            test_model_acc_score = accuracy_score(y_test,y_test_pred)
            test_model_f1_score = f1_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_acc_score

            return report
    except Exception as e:
        raise CustomException(e, sys)


