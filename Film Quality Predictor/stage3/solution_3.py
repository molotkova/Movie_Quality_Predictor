import pandas as pd

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

# solution of stage 3

clf_1 = LogisticRegression(solver='liblinear')
clf_1.fit(X_train, y_train)

accuracy_train = metrics.accuracy_score(y_train, clf_1.predict(X_train))
accuracy_test = metrics.accuracy_score(y_test, clf_1.predict(X_test))

# CHECKPOINT_3_1 - accuracy
print(round(accuracy_test, 5))
# >>> 0.8896

roc_auc_train = metrics.roc_auc_score(y_train, clf_1.predict_proba(X_train)[:, 1])
roc_auc_test = metrics.roc_auc_score(y_test, clf_1.predict_proba(X_test)[:, 1])

# CHECKPOINT_3_2 - AUC
print(round(roc_auc_test, 5))
# >>> 0.95845

# print("""0.8896
# 0.95845""")
