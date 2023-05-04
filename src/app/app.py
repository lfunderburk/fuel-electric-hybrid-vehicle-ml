from fastapi import FastAPI
import os
from dotenv import load_dotenv
from .app_utils import Prompter, init_database,  init_prompt, init_data
from pydantic import BaseModel

class Query(BaseModel):
    query: str

app = FastAPI()


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

    # Initialize data
    global df, sample_values
    df, sample_values = init_data()

    # Set up engine
    global engine
    engine = init_database(df)

@app.post("/search")
async def search(query: Query):
    
    # Initialize prompt
    sql_query = init_prompt(query, prompter, df, sample_values)
    print(sql_query)

    try:
        # Execute SQL query and fetch results
        with engine.connect() as connection:
            result = connection.execute(sql_query)
            rows = result.fetchall()

        # Convert rows to list of dicts for JSON response
        columns = result.keys()
        data = [dict(zip(columns, row)) for row in rows]
        return {"data": data, "sql_query": sql_query, "status": "success"}

    except Exception as e:
        return {"error": f"SQL query failed. {e}"}

