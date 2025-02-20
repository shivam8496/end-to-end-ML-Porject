import sys
import pandas as pd 
import numpy as np 
import os
from src.exception import CustomException
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

class Evaluate:
    def evaluate_models(self,X_train,Y_train,X_test,Y_test,models):
        try:
            report={}
            for i in range(len(list(models))):
                model=list(models.values())[i]
                model.fit(X_train,Y_train)
                Y_pred_train = model.predict(X_train)
                Y_pred_test = model.predict(X_test)
                train_score = r2_score(Y_train,Y_pred_train)
                test_score = r2_score(Y_test,Y_pred_test)
                report[list(models.keys())[i]]=test_score
            return report
        except Exception as e:
            raise CustomException(e,sys)
    
    def evaluate_models(self,X_train,Y_train,X_test,Y_test,models,params):
        try:
            report={}
            for i in range(len(list(models))):
                print(f"=========================================")
                model=list(models.values())[i]
                param=list(params.values())[i]
                model_name=list(models.keys())[i]
                print(f"model==>{model_name} and parameter==>{list(params.keys())[i]}")
                gs=GridSearchCV(model,param,cv=3)
                gs.fit(X_train,Y_train)
                print(f"Fitting into model==>{model_name}")
                model.set_params(**gs.best_params_)
                model.fit(X_train,Y_train)
                Y_pred_train = model.predict(X_train)
                Y_pred_test = model.predict(X_test)
                print(f"Predicting for  model==>{model_name} ")
                train_score = r2_score(Y_train,Y_pred_train)
                test_score = r2_score(Y_test,Y_pred_test)
                report[list(models.keys())[i]]=test_score
                print(f"\n")
                
            return report
        except Exception as e:
            raise CustomException(e,sys)

def load_data(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            model=dill.load(file_obj)
        return model
    except Exception as e:
        raise CustomException(e,sys)







