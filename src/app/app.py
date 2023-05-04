from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from .app_utils import Prompter, init_database, data_cleaner

def init_data():
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

    return df, sample_values

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
    prompter = Prompter(openai_api_key, "gpt-4")

    df, sample_values = init_data()

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
        return {"error": f"SQL query failed. {e}"}

    return {"data": data}
