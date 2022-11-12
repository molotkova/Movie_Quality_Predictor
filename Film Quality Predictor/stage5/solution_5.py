from sklearn.decomposition import TruncatedSVD

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# from stages 1-4
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

features_num = 103  # number of important features that we got on the prev stage

# solution of stage 4

clf_3 = LogisticRegression(solver='liblinear')
rounded_features = np.round(features_num, -2)

tsvd = TruncatedSVD(n_components=rounded_features, random_state=23)
X_train_pca = tsvd.fit_transform(X_train)
X_test_pca = tsvd.transform(X_test)

clf_3.fit(X_train_pca, y_train)

# CHECKPOINT_5_1 - accuracy score after SVD
print(round(metrics.accuracy_score(y_test, clf_3.predict(X_test_pca)), 5))
# >>> 0.85632

# CHECKPOINT_5_2 - AUC score after SVD
print(round(metrics.roc_auc_score(y_test, clf_3.predict_proba(X_test_pca)[:, 1]), 5))
# >>> 0.93701
