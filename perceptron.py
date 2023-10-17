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

# todo: maybe 0 should get -1
def sign(x):
    return 1 if x >= 0 else -1

def do_training(train_df, lr=1):
    x_df = train_df.loc[:, train_df.columns != 'Outcome']
    y_df = train_df['Outcome']
    
    d = x_df.shape[1]
    w = np.zeros((d+1,1))

    print(y_df[582])

    for index, row in x_df.iterrows():
        x = np.insert(np.array(row), 0, 1).reshape(-1, 1)
        # print('index', index)
        # print(x)
        pred = sign(np.dot(x.T, w))
        y = y_df[index]
        # print(y)
        if(pred != y):
            # print('prev w=', w)
            w += lr*y*x
            # print('after w=', w)

        # break
    return w


file_path = 'Dataset.csv'
df = do_preprocessings(file_path)

train_df=df.sample(frac=0.85,random_state=200)
test_df=df.drop(train_df.index)

print(do_training(train_df))