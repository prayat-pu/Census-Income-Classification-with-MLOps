import requests
import json

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

response = requests.post('https://census-income-classification-with-mlops.onrender.com/features/', data=json.dumps(test_data))
print('status code: ',response.status_code)
# print(response.json())