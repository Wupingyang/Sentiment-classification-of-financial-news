import pandas as pd
import numpy as np

train_data = pd.read_excel('E:\Programming/mo_fa_shi_bei/data/TRAIN.xlsx')
test_data = pd.read_excel('E:\Programming/mo_fa_shi_bei/data/TEST.xlsx')

train_data = train_data.replace([r'<.*?>', r'<img src="[a-zA-z]+://[^\s]*" alt="pic">'], '', regex=True)
test_data = test_data.replace([r'<.*?>', r'<img src="[a-zA-z]+://[^\s]*" alt="pic">'], '', regex=True)

print('saving train data...')
data_pos = train_data[train_data['score'] == 1]
data_neg = train_data[train_data['score'] == 2]
data_neu = train_data[train_data['score'] == 0]

train_data_pos = data_pos[['content']].reset_index(drop=True)
train_data_neg = data_neg[['content']].reset_index(drop=True)
train_data_neu = data_neu[['content']].reset_index(drop=True)

train_data_pos.to_csv('E:\Programming/mo_fa_shi_bei/data/train_data_pos.csv')
train_data_neg.to_csv('E:\Programming/mo_fa_shi_bei/data/train_data_neg.csv')
train_data_neu.to_csv('E:\Programming/mo_fa_shi_bei/data/train_data_neu.csv')

print('saving test data...')

test_data = test_data[['content']].reset_index(drop=True)

test_data.to_csv('E:\Programming/mo_fa_shi_bei/data/test_data.csv')

