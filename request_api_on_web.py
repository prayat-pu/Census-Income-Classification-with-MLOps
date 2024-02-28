import requests
import json

test_data1 = {
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

response1 = requests.post(
    'https://census-income-classification-with-mlops.onrender.com/predict/',
    data=json.dumps(test_data1))
print('status code: ', response1.status_code)
print(response1.json())

print('-----another test case.-----')
test_data2 = {
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

response2 = requests.post(
    'https://census-income-classification-with-mlops.onrender.com/predict/',
    data=json.dumps(test_data2))
print('status code: ', response2.status_code)
print(response2.json())
