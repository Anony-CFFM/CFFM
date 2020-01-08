import pandas as pd


validation = pd.read_csv('ml-tag.validation.libfm', encoding='latin-1', sep=' ')
test = pd.read_csv('ml-tag.test.libfm', encoding='latin-1', sep=' ')
train = pd.read_csv('ml-tag.train.libfm', encoding='latin-1', sep=' ')

print(validation.shape[0])
print(test.shape[0])
print(train.shape[0])
print('all:',(validation.shape[0]+test.shape[0]+train.shape[0]))