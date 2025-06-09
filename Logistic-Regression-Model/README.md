This folder contains two models for essay score analysis using TF-IDF vectorization:

1. Regression Model: Predicts the continuous score value.

2. Classification Model: Categorizes essay scores into predefined bands.

# Structure
`regression_model.py`: Uses models like Ridge or LinearRegression to predict exact scores.

`classification_model.py`: Uses models like LogisticRegression to classify scores into bands (<5, 5, 6, 7, 8+).

# Features
Text Preprocessing: Merges question and essay text.

TF-IDF Vectorization: Configurable max_features and n_gram_range.

Model Training: Flexible base_model input.

Evaluation:

Regression: MAE, MSE, RMSE, R², and scatterplot visualization.

Classification: Accuracy, classification report, and confusion matrix.

# Usage
Instantiate either class with a scikit-learn model, TF-IDF params, and data path. Example:

```python

from sklearn.linear_model import Ridge, LogisticRegression
from regression_model import RegressionModel
from classification_model import ClassificationModel

reg_model = RegressionModel(Ridge(), 5000, (1, 2), "../data/filter_dataset.csv")
reg_model.train()
reg_model.evaluate()

clf_model = ClassificationModel(LogisticRegression(), 5000, (1, 2), "../data/filter_dataset.csv")
clf_model.train()
clf_model.evaluate()
```