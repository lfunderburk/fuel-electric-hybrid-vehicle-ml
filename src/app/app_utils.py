import pandas as pd
import os
import openai
from sqlalchemy.engine import create_engine
from pathlib import Path

class Prompter:
    def __init__(self, api_key, gpt_model):
        if not api_key:
            raise Exception("Please provide the OpenAI API key")


        openai.api_key = os.environ.get("OPENAI_API_KEY")

        self.gpt_model = gpt_model
    
    def prompt_model_return(self, messages: list):
        response = openai.ChatCompletion.create(model=self.gpt_model, 
                                                messages=messages,
                                                temperature=0.2)
        return response["choices"][0]["message"]["content"]
    
def data_cleaner(df):
    

    # Replace the characters '.', '/', '(' and ')'with '_per_' all entries
    df.columns = df.columns.str.replace('.', '_', regex=True)
    df.columns = df.columns.str.replace('/', '_per_', regex=True)
    df.columns = df.columns.str.replace('(', '_', regex=True)
    df.columns = df.columns.str.replace(')', '_', regex=True)

    # drop column hybrid_in_fuel	hybrid_in_electric	aggregate_levels	vehicle_type_cat
    df = df.drop(['hybrid_in_fuel', 'hybrid_in_electric', 'aggregate_levels','transmission_','fuel_type'], axis=1)

    return df

def init_database(df):
    # Set up engine
    engine = create_engine("sqlite://")
    df.to_sql("vehicleDB", engine)

    return engine

