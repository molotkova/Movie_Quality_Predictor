import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# from stages 1-2
data = pd.read_csv('../Data/dataset.csv', sep=',')

data = data[(data['rating'] > 7) | (data['rating'] < 5)]  # filtering of data
data['label'] = bool
data.loc[(data.rating > 7), 'label'] = '1'
data.loc[(data.rating < 5), 'label'] = '0'
data.drop(columns='rating', axis=1, inplace=True)

# splitting matrices into random train and test subsets
texts_train, texts_test, y_train, y_test = train_test_split(data.review.values,
                                                            data.label.values,
                                                            random_state=23)
vect = TfidfVectorizer(sublinear_tf=True)
X_train = vect.fit_transform(texts_train)
X_test = vect.transform(texts_test)

# solution of stage 4

clf_2 = LogisticRegression(solver='liblinear')
clf_2.C = 0.15
clf_2.penalty = 'l1'

# print(clf_2)
clf_2.fit(X_train, y_train)

# CHECKPOINT_4_1 - accuracy score after LASSO
print(round(metrics.accuracy_score(y_test, clf_2.predict(X_test)), 5))
# >>> 0.81024

# CHECKPOINT_4_2 - AUC score after LASSO
print(round(metrics.roc_auc_score(y_test, clf_2.predict_proba(X_test)[:, 1]), 5))
# >>> 0.89102

# CHECKPOINT_4_3 - number of features after LASSO
features_num = np.sum(np.abs(clf_2.coef_) > 0.0001)
print(features_num)
# >>> 103




