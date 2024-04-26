import pickle
import os

from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from replace_dicts import *

# Define a Pydantic model for the request


class BankFeatures(BaseModel):
    age: int
    job: str
    martial: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str


class instanceList(BaseModel):
    instances: List[BankFeatures]


with open(os.environ['MODEL_PATH'], 'rb') as file:
    model = pickle.load(file)

app = FastAPI()


@app.get("/status")
async def get_status():
    return {"status": "OK"}


@app.post("/predict")
async def predict_function(data: instanceList):
    predictions = []
    for features in data.instances:
        # Prepare the input data for prediction
        input_data = [[
            features.age,
            job_replace_dict[features.job],
            martial_replace_dict[features.martial],
            education_replace_dict[features.education],
            binary_replace_dict[features.default],
            features.balance,
            binary_replace_dict[features.housing],
            binary_replace_dict[features.loan],
            contact_replace_dict[features.contact],
            features.day,
            month_replace_dict[features.month],
            features.duration,
            features.campaign,
            features.pdays,
            features.previous,
            poutcome_replace_dict[features.poutcome]
        ]]

        # Make a prediction
        prediction = model.predict(input_data)

        # Map the predicted class to the corresponding output name
        output_names = ["no", "yes"]
        predicted_outcome = output_names[prediction[0]]
        predictions.append(predicted_outcome)

    return {"predictions": predictions}
