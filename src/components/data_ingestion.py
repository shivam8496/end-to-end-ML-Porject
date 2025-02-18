import os
import sys
import pandas as pd 

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformer
from src.components.data_transformation import DataTranformationConfig
from src.components.model_trainer import ModelTrainerConfig , ModelTrainer

from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or components")
        try:
            df=pd.read_csv( os.path.join("notebook", "data", "StudentsPerformance.csv") )

            logging.info("Ingested DataSet as data frame")

            # Trainging data folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Testing Data folder
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)

            # Raw Data folder and saving it
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split Initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            # train_set,test_set = train_test_split(df,train_size=0.8,random_state=42)

            #saving training data
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            #Saving testing data
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("DataIngestion completed!!!!! :)")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj1=DataIngestion()
    train,test=obj1.initiate_data_ingestion()
    obj2=DataTransformer()
    train_arr,test_arr,_ = obj2.initiate_data_tranformation(train,test)
    modelTrainer=ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_arr,test_arr))