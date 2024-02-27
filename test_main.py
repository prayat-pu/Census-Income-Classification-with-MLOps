from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_path():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {'greeting': "Welcome to the Census classification API"}


def test_post_lessthan50k_case():
    test_data = {
        'age': 34,
        'workclass': ' Private',
        'fnlgt': 133503,
        'education': ' Some-college',
        'education_num': 10,
        'marital_status': ' Divorced',
        'occupation': ' Transport-moving',
        'relationship': ' Not-in-family',
        'race': ' White',
        'sex': ' Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': ' United-States'
    }

    r = client.post('/features/',
                    json=test_data)

    assert r.status_code == 200
    assert (r.json() == 'the prediction of this features: <=50k')


def test_post_greaterthan50k_case():
    test_data = {
        'age': 37,
        'workclass': ' State-gov',
        'fnlgt': 160402,
        'education': ' Bachelors',
        'education_num': 13,
        'marital_status': ' Married-civ-spouse',
        'occupation': ' Prof-specialty',
        'relationship': ' Husband',
        'race': ' White',
        'sex': ' Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 55,
        'native_country': ' United-States'
    }

    r = client.post('/features/',
                    json=test_data)

    assert r.status_code == 200
    assert (r.json() == 'the prediction of this features: >50k')
