import sys
import pandas as pd 
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from  src.exception import CustomException

from src.utils import load_data
from src.components.data_transformation import DataTranformationConfig
from src.components.model_trainer import ModelTrainerConfig

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path=DataTranformationConfig().preprocessor_file_path
            model_path=ModelTrainerConfig().trained_model_file_path
            print("Before Loading")
            preprocessor=load_data(file_path=preprocessor_path)
            model = load_data(file_path=model_path)
            scaled_data = preprocessor.transform(features)
            print("After Loading")
            predictions = model.predict(scaled_data)
            return predictions
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,
            gender:str,
            race_ethnicity:str,
            parental_level_of_education:str,
            lunch:str,
            test_preparation_course:str,
            reading_score:int,
            writing_score:int,
            ):
        self.gender:str=gender
        self.race_ethnicity:str=race_ethnicity
        self.parental_level_of_education:str=parental_level_of_education
        self.lunch:str=lunch
        self.test_preparation_course:str=test_preparation_course
        self.reading_score:int=reading_score
        self.writing_score:int=writing_score
    
    def get_data_as_dataframe(self):
        try:
            data={
            "gender":[self.gender],
            "race/ethnicity":[self.race_ethnicity],
            "parental level of education":[self.parental_level_of_education],
            "lunch":[self.lunch],
            "test preparation course":[self.test_preparation_course],
            "reading score":[self.reading_score],
            "writing score":[self.writing_score]
            }
            return pd.DataFrame(data)
        
        except Exception as e:
            raise ConnectionRefusedError(e,sys)
        
# if __name__=="__main__":
#     data=CustomData(gender="female",
#             race_ethnicity="group B",
#             parental_level_of_education="bachelor's degree",
#             lunch="standard",
#             test_preparation_course="none",
#             reading_score=72,
#             writing_score=74).get_data_as_dataframe()
#     predictions=PredictPipeline()
#     print(predictions.predict(data))
