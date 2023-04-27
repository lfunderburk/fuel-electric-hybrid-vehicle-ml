from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pathlib import Path
import openai
from sqlalchemy.engine import create_engine
import pandas as pd

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
    df.columns = df.columns.str.replace('.', '_')

    # Replace the character '/' with '_per_' all entries
    df.columns = df.columns.str.replace('/', '_per_')

    df.columns = df.columns.str.replace('(', '_')
    
    df.columns = df.columns.str.replace(')', '_')

    # drop column hybrid_in_fuel	hybrid_in_electric	aggregate_levels	vehicle_type_cat
    df = df.drop(['hybrid_in_fuel', 'hybrid_in_electric', 'aggregate_levels','transmission_','fuel_type'], axis=1)

    return df

def init_database(df):
    # Set up engine
    engine = create_engine("sqlite://")
    df.to_sql("vehicleDB", engine)

    return engine

app = FastAPI()

class Query(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Welcome! Please send a POST request to /search with a JSON body containing a 'query' key and a natural language query as the value. Visit /docs for more "}

@app.on_event("startup")
async def startup_event():
    # Load the .env file
    load_dotenv(".env")

    # Get OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # Initialize prompter
    global prompter
    prompter = Prompter(openai_api_key, "gpt-3.5-turbo")

    # Set the path to the raw data
    # Convert the current working directory to a Path object
    script_dir = Path(os.getcwd())
    predicted_data_path = script_dir / 'data' / 'predicted-data' / 'vehicle_data_with_clusters.csv'
    
    # Load the CSV file into a DataFrame
    dirty_df = pd.read_csv(predicted_data_path)
    global df
    df = data_cleaner(dirty_df)
    global sample_values
    sample_values = {df.columns[i]: df.values[0][i] for i in range(len(df.columns))}

    # Set up engine
    global engine
    engine = init_database(df)

@app.post("/search")
async def search(query: Query):
    # Generate SQL query
    datagen_prompts = [
        {"role" : "system", "content" : "You are a data analyst specializing in SQL, you are presented with a natural language query, and you form queries to answer questions about the data."},
        {"role" : "user", "content" : f"Please generate 1 SQL queries for data with columns {', '.join(df.columns)} and sample values {sample_values}. \
                                        The table is called 'vehicleDB'. Use the natural language query {query.query}"},
    ]

    result = prompter.prompt_model_return(datagen_prompts)
    print(result)
    sql_query = result.split("\n\n")[0]

    try:
        # Execute SQL query and fetch results
        with engine.connect() as connection:
            result = connection.execute(sql_query)
            rows = result.fetchall()

        # Convert rows to list of dicts for JSON response
        columns = result.keys()
        data = [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        print(e)
        return {"error": "SQL query failed"}

    return {"data": data}
