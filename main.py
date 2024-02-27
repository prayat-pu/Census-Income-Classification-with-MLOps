import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from ml.model import inference
from ml.data import process_data


class Data(BaseModel):
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


app = FastAPI(
    title='Census Classification API',
    description='An aPI that demonstrates checking\
          the inference of the census classification',
    version='1.0.0')


@app.get('/')
async def greeting():
    return {'greeting': "Welcome to the Census classification API"}


@app.post('/features/')
async def create_item_for_model_inference(data: Data):

    model_filepath = './model/trained_model.pkl'
    with open(model_filepath, 'rb') as file:
        model = pickle.load(file)

    encoder_filepath = './model/encoder.pkl'
    with open(encoder_filepath, 'rb') as file:
        encoder = pickle.load(file)

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
