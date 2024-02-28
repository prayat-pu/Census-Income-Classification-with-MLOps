import pickle
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import pandas as pd
from ml.model import inference
from ml.data import process_data
from contextlib import asynccontextmanager

def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")

class Data(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: hyphen_to_underscore(field_name)
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model,encoder,cat_features
    # Load the ML model, and other
    model = pickle.load(open("./model/trained_model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    yield

    del model
    del encoder
    del cat_features

app = FastAPI(
    title='Census Classification API',
    description='An aPI that demonstrates checking\
          the inference of the census classification',
    version='1.0.0',
    lifespan=lifespan)


@app.get('/')
async def greeting():
    return {'greeting': "Welcome to the Census classification API"}


@app.post('/predict/')
async def create_item_for_model_inference(data: Data):

    data_dict = {
        'age': [data.age],
        'workclass': [data.workclass],
        'fnlgt': [data.fnlgt],
        'education': [data.education],
        'education-num': [data.education_num],
        'marital-status': [data.marital_status],
        'occupation': [data.occupation],
        'relationship': [data.relationship],
        'race': [data.race],
        'sex': [data.sex],
        'capital-gain': [data.capital_gain],
        'capital-loss': [data.capital_loss],
        'hours-per-week': [data.hours_per_week],
        'native-country': [data.native_country]
    }

    new_df = pd.DataFrame(data_dict)
    x_test, _, _, _ = process_data(
        new_df, categorical_features=cat_features,
        training=False, encoder=encoder)
    prediction = inference(model, x_test)

    if len(prediction) == 1:
        if prediction[0] == 0:
            predict_class = '<=50k'
        else:
            predict_class = '>50k'
        return f'the prediction of this features: {predict_class}'
    elif len(prediction) > 1:
        return str(prediction)
