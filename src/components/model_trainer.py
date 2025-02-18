import os 
import sys


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree  import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import (save_object,evaluate_models)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and testing input data")
            X_train,Y_train,X_test,Y_test=(
             train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(X_train,Y_train,X_test,Y_test,models)
                                            
            best_model_score=max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if(best_model_score<0.6):
                print(best_model_score)
                raise Exception("No Best Model Found")
            
            logging.info("Best Model Found for Both training and testing DataSets")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            
            r2_square=r2_score(Y_test,predicted)
            logging.info("model saved succesfully")
            return r2_square,best_model_name
        
        except Exception as e:
            raise CustomException(e,sys)


# import os
# import sys
# from dataclasses import dataclass

# from catboost import CatBoostRegressor
# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor,
# )
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

# from src.exception import CustomException
# from src.logger import logging

# from src.utils import save_object,evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path=os.path.join("artifacts","model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config=ModelTrainerConfig()


#     def initiate_model_trainer(self,train_array,test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train,y_train,X_test,y_test=(
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )
#             models = {
#                 "Random Forest": RandomForestRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "Linear Regression": LinearRegression(),
#                 "XGBRegressor": XGBRegressor(),
#                 "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#                 "AdaBoost Regressor": AdaBoostRegressor(),
#             }
           
#             model_report:dict=evaluate_models(X_train=X_train,Y_train=y_train,X_test=X_test,Y_test=y_test,
#                                              models=models)
            
#             ## To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             ## To get best model name from dict

#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score<0.6:
#                 raise CustomException("No best model found")
#             logging.info(f"Best found model on both training and testing dataset")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted=best_model.predict(X_test)

#             r2_square = r2_score(y_test, predicted)
#             return r2_square , best_model_name 
            
#         except Exception as e:
#             raise CustomException(e,sys)