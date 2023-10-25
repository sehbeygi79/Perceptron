import numpy as np
import pandas as pd


def do_preprocessings(file_path):
    df = pd.read_csv(file_path)

    # shuffle
    # df = df.sample(frac=1).reset_index(drop=True)
    # remove corrupted data
    df = df.loc[(df['BMI'] != 0) & (df['Glucose'] != 0)
                & (df['BloodPressure'] != 0)]
    x_df = df.loc[:, df.columns != 'Outcome']
    y_df = df['Outcome']
    y_df.replace(to_replace={True: 1, False: -1}, inplace=True)
    # normalize
    x_df = (x_df-x_df.min())/(x_df.max()-x_df.min())

    df.loc[:, df.columns != 'Outcome'] = x_df
    df['Outcome'] = y_df
    return df


def train(train_df, lr=1, epoch=1):
    x_df = train_df.loc[:, train_df.columns != 'Outcome']
    y_df = train_df['Outcome']

    d = x_df.shape[1]
    w = np.zeros((d+1, 1))
    
    for i in range(epoch):
        for index, row in x_df.iterrows():
            x = np.insert(np.array(row), 0, 1).reshape(-1, 1)
            pred = np.sign(np.dot(x.T, w))
            y = y_df.loc[index]
            if(pred != y):
                w += lr*y*x

    return w


def predict(X, w):
    # todo: assertion for w.shape == (x_df.shape[1]+1, 1)
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((ones_column, X))
    return np.sign(np.dot(X, w))


def calc_acc(y, y_hat):
    correct_preds = np.sum(y == y_hat)
    return (correct_preds / len(y)) * 100
    


file_path = 'Dataset.csv'
df = do_preprocessings(file_path)

seed = 37
train_df = df.sample(frac=0.85, random_state=seed)
test_df = df.drop(train_df.index)
test_x = test_df.loc[:, test_df.columns != 'Outcome'].values
test_y = test_df['Outcome'].values.reshape(-1, 1)

weights = train(train_df, epoch=1)

test_y_hat = predict(test_x, weights)
print(calc_acc(test_y, test_y_hat))
