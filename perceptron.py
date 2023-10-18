import numpy as np
import pandas as pd

def do_preprocessings(file_path):
    df = pd.read_csv(file_path)

    # shuffle
    # df = df.sample(frac=1).reset_index(drop=True)
    # remove corrupted data
    df = df.loc[(df['BMI'] != 0) & (df['Glucose'] != 0) & (df['BloodPressure'] != 0)]
    x_df = df.loc[:, df.columns != 'Outcome']
    y_df = df['Outcome']
    y_df.replace(to_replace={True:1, False:-1}, inplace=True)
    # normalize
    x_df = (x_df-x_df.min())/(x_df.max()-x_df.min())
    
    df.loc[:, df.columns != 'Outcome'] = x_df
    df['Outcome'] = y_df
    return df

def do_training(train_df, lr=1):
    x_df = train_df.loc[:, train_df.columns != 'Outcome']
    y_df = train_df['Outcome']
    
    d = x_df.shape[1]
    w = np.zeros((d+1,1))

    for index, row in x_df.iterrows():
        x = np.insert(np.array(row), 0, 1).reshape(-1, 1)
        pred = np.sign(np.dot(x.T, w))
        y = y_df.loc[index]
        if(pred != y):
            w += lr*y*x

    return w

def do_test(test_df, w):
    # todo: assertion for w.shape == (x_df.shape[1]+1, 1)

    x_df = test_df.loc[:, test_df.columns != 'Outcome']
    X = x_df.values
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((ones_column, X))

    y_df = test_df['Outcome']
    Y = y_df.values.reshape(-1, 1)

    pred = np.sign(np.dot(X, w))

    correct_preds = np.sum(pred == Y)
    accuracy = (correct_preds / len(Y)) * 100
    print(accuracy)
    


file_path = 'Dataset.csv'
df = do_preprocessings(file_path)

seed = 37
train_df=df.sample(frac=0.85,random_state=seed)
test_df=df.drop(train_df.index)

weights = do_training(train_df)

do_test(test_df, weights)
