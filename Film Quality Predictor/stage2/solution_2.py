import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# from stage 1
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

# CHECKPOINT_2_1 - number of features after transformation (vectorization)
print(X_train.shape[1])
# >>> 66648
