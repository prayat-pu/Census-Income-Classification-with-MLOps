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
        'education-num': 10,
        'marital-status': ' Divorced',
        'occupation': ' Transport-moving',
        'relationship': ' Not-in-family',
        'race': ' White',
        'sex': ' Male',
        'capital-gain': 2174,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': ' United-States'
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
        'education-num': 13,
        'marital-status': ' Married-civ-spouse',
        'occupation': ' Prof-specialty',
        'relationship': ' Husband',
        'race': ' White',
        'sex': ' Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 55,
        'native-country': ' United-States'
    }

    r = client.post('/features/',
                    json=test_data)

    assert r.status_code == 200
    assert (r.json() == 'the prediction of this features: >50k')
