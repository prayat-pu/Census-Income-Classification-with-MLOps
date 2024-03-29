import pickle
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
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

    age: int = Field(example=45)
    workclass: str = Field(example='State-gov')
    fnlgt: int = Field(example=2334)
    education: str = Field(example='Bachelors')
    education_num: int = Field(example=13)
    marital_status: str = Field(example='Never-married')
    occupation: str = Field(example='Prof-specialty')
    relationship: str = Field(example='Wife')
    race: str = Field(example='Black')
    sex: str = Field(example='Female')
    capital_gain: int = Field(example=2174)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=60)
    native_country: str = Field(example='Cuba')

loads = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model, and other
    loads['model'] = pickle.load(open("./model/trained_model.pkl", "rb"))
    loads['encoder'] = pickle.load(open("./model/encoder.pkl", "rb"))
    loads['cat_features']  = [
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

    loads.clear()

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
        new_df, categorical_features=loads['cat_features'],
        training=False, encoder=loads['encoder'])
    prediction = inference(loads['model'], x_test)

    if len(prediction) == 1:
        if prediction[0] == 0:
            predict_class = '<=50k'
        else:
            predict_class = '>50k'
        return f'the prediction of this features: {predict_class}'
    elif len(prediction) > 1:
        return str(prediction)
