# Model Card of Census Classification

## Model Details
This model is tailored for binary classification purposes. An output of 0 from the model suggests that it predicts the individual's income to be below $50,000, whereas an output of 1 suggests a prediction of income exceeding $50,000. In this iteration, the model employs a random forest algorithm, a widely recognized method suitable for structured data. At present, the model's parameters are set to the default values provided by scikit-learn.

## Intended Use
This model is utilized to predict an individual's annual income based on 14 distinct features.

## Training Data and Evaluation Data
The model is trained and evaluated using data from the census dataset, available at https://archive.ics.uci.edu/dataset/20/census+income. The dataset is divided, allocating 80% for training purposes and 20% for evaluation.

## Metrics
In this research, the evaluation metrics employed are those commonly used in classification tasks: precision, recall, and the F1 score. These metrics are widely recognized for assessing the performance of classification models.

### Performance Metrics of the Current Version
- Precision: 0.73
- Recall: 0.63
- F1 Score: 0.67

## Ethical Considerations
Please exercise caution when using this model for predictions. It is important to obtain consent from individuals prior to making predictions about them and to handle their sensitive information with care.


## Caveats and Recommendations
The accuracy of the model's current version is not guaranteed, so it should not be used inappropriately or in contexts where high accuracy is critical. In the future, enhancing the model's performance could be achieved through modifying its algorithm or adjusting its hyperparameters. Furthermore, incorporating additional features may also lead to improvements.

