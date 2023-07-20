import os
import sys
from Adult_census_income_predn.logging.logger import logging
from Adult_census_income_predn.exceptions.exception import CustomException
import pandas as pd

from Adult_census_income_predn.components.data_ingestion import DataIngestion
from Adult_census_income_predn.components.data_transformation import DataTransformation
from Adult_census_income_predn.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)



