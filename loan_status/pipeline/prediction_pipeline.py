import os
import sys

import numpy as np
import pandas as pd
from loan_status.entity.config_entity import LoanStatusPredictorConfig
from loan_status.entity.s3_estimator import LoanStatusEstimator
from loan_status.exception import LoanStatusException
from loan_status.logger import logging
from loan_status.utils.main_utils import read_yaml_file
from pandas import DataFrame


class LoanStatusData:
    def __init__(self,
                Gender,
                Married,
                Dependents,
                Education,
                Self_Employed,
                ApplicantIncome,
                CoapplicantIncome,
                LoanAmount,
                Loan_Amount_Term,
                Credit_History,
                Property_Area

                ):
        """
        LoanStatus Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Gender = Gender
            self.Married = Married
            self.Dependents = Dependents
            self.Education = Education
            self.Self_Employed = Self_Employed
            self.ApplicantIncome = ApplicantIncome
            self.CoapplicantIncome = CoapplicantIncome
            self.LoanAmount = LoanAmount
            self.Loan_Amount_Term = Loan_Amount_Term
            self.Credit_History = Credit_History
            self.Property_Area = Property_Area


        except Exception as e:
            raise LoanStatusException(e, sys) from e

    def get_loanstatus_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from LoanStatusData class input
        """
        try:
            
            loanstatus_input_dict = self.get_loanstatus_data_as_dict()
            return DataFrame(loanstatus_input_dict)
        
        except Exception as e:
            raise LoanStatusException(e, sys) from e


    def get_loanstatus_data_as_dict(self):
        """
        This function returns a dictionary from LoanStatusData class input 
        """
        logging.info("Entered get_loanstatus_data_as_dict method as LoanStatusData class")

        try:
            input_data = {
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "ApplicantIncome": [self.ApplicantIncome],
                "CoapplicantIncome": [self.CoapplicantIncome],
                "LoanAmount": [self.LoanAmount],
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Credit_History": [self.Credit_History],
                "Property_Area": [self.Property_Area],
            }

            logging.info("Created LoanStatus data dict")

            logging.info("Exited get_loanstatus_data_as_dict method as LoanStatus class")

            return input_data

        except Exception as e:
            raise LoanStatusException(e, sys) from e

class LoanStatusClassifier:
    def __init__(self,prediction_pipeline_config: LoanStatusPredictorConfig = LoanStatusPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise LoanStatusException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of LoanStatusClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of LoanStatusClassifier class")
            model = LoanStatusEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise LoanStatusException(e, sys)