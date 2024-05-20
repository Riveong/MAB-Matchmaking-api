from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
import pandas as pd
from func import master_function
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json
import os


app = FastAPI()

origins = ["*"]

PORT = 8000
HOST = '0.0.0.0'

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_path = '/data.csv'

class Answers(BaseModel):
    answers: list[int]
    age: int
    gender:int

class UserData(BaseModel):
    nama: str
    umur: int
    gender: int
    epsilon_greedy_id: int
    thompson_sampling_id: int
    user_input: List[int]
    rating: List[Optional[int]]
    reward: List[int]


def append_list_as_first_row(file_path, new_row_data):
    # Read the existing CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Create a new DataFrame from the new row data
    new_row_df = pd.DataFrame([new_row_data], columns=df.columns)
    
    # Concatenate the new row DataFrame with the existing DataFrame
    # Place it at the top using concat and ignore_index=True
    updated_df = pd.concat([new_row_df, df], ignore_index=True)
    
    # Write the updated DataFrame back to the CSV
    updated_df.to_csv(file_path, index=False)

def search_in_column(file_path, column_name, search_value):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Perform the search
    result_df = df[df[column_name] == search_value]
    
    return result_df

def retrieve_row_data(file_path, row_index):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure the row index is within the range of the DataFrame's rows
    if row_index >= len(df) or row_index < 0:
        raise IndexError("Row index is out of bounds. Please provide a valid index.")
    
    # Retrieve the row data
    row_data = df.iloc[row_index]  # Use .iloc to retrieve row data
    
    return row_data


def save_data_to_file(data):
    filename = "hasil.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r+", encoding='utf-8') as file:
                existing_data = json.load(file)
                existing_data.append(data)
                file.seek(0)
                json.dump(existing_data, file, indent=4)
        except json.JSONDecodeError:
            # If JSON is empty or corrupted, start a new list
            with open(filename, "w", encoding='utf-8') as file:
                json.dump([data], file, indent=4)
    else:
        with open(filename, "w", encoding='utf-8') as file:
            json.dump([data], file, indent=4)

@app.post("/submit-data/")
async def submit_data(user_data: UserData):
    save_data_to_file(user_data.dict())
    return {"message": "Data saved successfully"}

@app.get("/search_similarity/{id}")
async def search(id: int):
    try:
        # Read and process the data
        data = search_in_column('preprocessed_data.csv', 'match_id', id)
        # Convert DataFrame to a list of lists, then flatten it
        data_list = data.values.tolist()
        flat_list = [item for sublist in data_list for item in sublist]
        return {
            "status": "success",
            "data": flat_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/search_data/{id}")
async def search(id: int):
    try:
        data = retrieve_row_data('data.csv', id)
        return {"status": "success", "data": data.tolist()}  # Ensure data is serializable
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.post("/run_matchmaking/")
async def run_matchmaking(answers: Answers):
    try:
        if len(answers.answers) != 40:
            raise HTTPException(status_code=505, detail="25 len req")
        answers.answers.append(answers.age)
        answers.answers.append(answers.gender)
        append_list_as_first_row('data.csv', answers.answers)
        results = master_function('data.csv')
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"Hello": "World"}


# Function to read data from file
def read_data_from_file():
    filename = "hasil.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="JSON file is corrupted")
    else:
        raise HTTPException(status_code=404, detail="JSON file not found")

# Define the API endpoint to return all data
@app.get("/get-all-data/")
async def get_all_data():
    data = read_data_from_file()
    return data

if __name__ == "__main__":
    uvicorn.run(app, host = HOST, port = PORT, reload = True)
