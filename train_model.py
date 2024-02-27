# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
import pandas as pd
import pickle


def measuring_performance_on_sclicing_cencus(df, cat_features, scling_features,
                                             model, encoder, lb, label):
    """ Function for measuring performance on slices of the cencus dataset.

     Inputs
    ------
    Model : ???
        Trained machine learning model.
    df : np.array
        Data used for prediction.
    y: np.array
        label to measure performance of the model.
    features: str
        features name for slicing the dataset.
    -------
    """

    with open('./slice_output.txt','w') as f:
        f.write(f'Performance on slicing features: {scling_features} feaure')
        f.write('\n')
    

        for ele in df[scling_features].unique():
            data = df[df[scling_features] == ele]

            X_test, y_test, _, _ = process_data(
                data, categorical_features=cat_features,
                label="salary", training=False, encoder=encoder, lb=lb
            )

            y_pred = inference(model, X_test)

            precision, recall, f1 = compute_model_metrics(y_test, y_pred)
            f.write('-' * 20)
            f.write('\n')
            f.write(f'Features Value: {ele}\n')
            f.write(f'total_data: {len(data)}\n')
            f.write(f'Precision: {precision}\n')
            f.write(f'recall: {recall}\n')
            f.write(f'f1: {f1}\n')
            f.write('-' * 20)
            f.write('\n')


# Add code to load in the data.
data = pd.read_csv('./data/census.csv')
# add some preprocessing from eda -> we found 23 duplicated rows
data.drop_duplicates(inplace=True)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)


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


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print('Performance for the whole test dataset')
print(f'precision: {precision}, recall: {recall}, f1: {fbeta}')

# test performance on slicing selected features. -> scling_features =
# 'education'
slicing_flag = True
if slicing_flag:
    measuring_performance_on_sclicing_cencus(
        test, cat_features, 'education', model, encoder, lb, label='salary')


# save model and encoder
pickle.dump(model, open('./model/trained_model.pkl', 'wb'))
pickle.dump(encoder, open('./model/encoder.pkl', 'wb'))
pickle.dump(lb, open('./model/lb.pkl', 'wb'))
