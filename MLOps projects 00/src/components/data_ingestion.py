import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# split the data and save it to artifacts folder
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv') 
    test_data_path: str = os.path.join('artifacts','test.csv') 
    row_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('data\Heart Attack Data Set.csv')
            logging.info('Read the data as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.row_data_path,index=False, header=True)

            logging.info("Train test split initited")
            trian_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            trian_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

            logging.info("Ingestion of the data iss completed")

            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()