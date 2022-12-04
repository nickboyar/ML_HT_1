import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float


class Items(BaseModel):
    objects: List[Item]




@app.post("/predict_item")
def predict_item(item: Item) -> float:
    lr_ridge = joblib.load("weights.pickle")
    d = item.dict().copy()

    input_data = pd.DataFrame([d], columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])

    X_train = input_data.drop(columns=['name', 'fuel', 'seller_type', 'transmission', 'owner'])

    return lr_ridge.predict(X_train)[0]

