import logging
import os
import pickle
from typing import List, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class HeartModel(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=80, max_items=80)]
    features: List[str]


class Response(BaseModel):
    id: str
    price: float



def make_predict(
        data: List, features: List[str], model: Pipeline,
) -> List[Response]:
    data = pd.DataFrame(data, columns=features)
    ids = [int(x) for x in data["Id"]]
    predicts = model.predict(data)

    return [
        Response(id=id_, price=float(price)) for id_, price in zip(ids, predicts)
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    model_path = "model.pkl"
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/healz")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=List[Response])
def predict(request: HeartModel):
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))