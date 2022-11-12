import pandas as pd
import os
import requests

# Data downloading script

########
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if ('dataset.csv' not in os.listdir('../Data')):
    print('Dataset loading.')
    url = "https://www.dropbox.com/s/0sj7tz08sgcbxmh/large_movie_review_dataset.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/dataset.csv', 'wb').write(r.content)
    print('Loaded.')
# The dataset is saved to `Data` directory
########

# write your code here

# Reference solution.
data = pd.read_csv('../Data/dataset.csv', sep=',')

# CHECKPOINT_1_1 - number of rows
print(data.shape[0])
# >>> 32086

data = data[(data['rating'] > 7) | (data['rating'] < 5)]  # filtering of data


# CHECKPOINT_1_2 - number of rows after filtering
print(data.shape[0])
# >>> 25000

data['label'] = bool
data.loc[(data.rating > 7), 'label'] = '1'
data.loc[(data.rating < 5), 'label'] = '0'

data.drop(columns='rating', axis=1, inplace=True)

# CHECKPOINT_1_3 - proportions of classes
print(data['label'].value_counts()[0] / data.shape[0])
# >>> 0.5
