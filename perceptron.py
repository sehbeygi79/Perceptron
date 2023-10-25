import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, file_path, seed=37):
        df = self.__do_preprocessings(pd.read_csv(file_path))

        self.train_df = df.sample(frac=0.85, random_state=seed)
        test_df = df.drop(self.train_df.index)
        self.test_x = test_df.loc[:, test_df.columns != 'Outcome'].values
        self.test_y = test_df['Outcome'].values.reshape(-1, 1)

    def __do_preprocessings(self, df):
        # shuffle
        # df = df.sample(frac=1).reset_index(drop=True)
        # remove corrupted data
        df = df.loc[(df['BMI'] != 0) & (df['Glucose'] != 0)
                    & (df['BloodPressure'] != 0)]
        x_df = df.loc[:, df.columns != 'Outcome']
        y_df = df.loc[:, df.columns == 'Outcome'].replace(
            to_replace={True: 1, False: -1})
        # normalize
        x_df = (x_df-x_df.min())/(x_df.max()-x_df.min())

        return pd.concat([x_df, y_df], axis=1, join='inner')

    def train(self, lr=1, epoch=1, evaluate=False):
        x_df = self.train_df.loc[:, self.train_df.columns != 'Outcome']
        y_df = self.train_df['Outcome']

        d = x_df.shape[1]
        w = np.zeros((d+1, 1))

        for i in range(epoch):
            for index, row in x_df.iterrows():
                x = np.insert(np.array(row), 0, 1).reshape(-1, 1)
                pred = np.sign(np.dot(x.T, w))
                y = y_df.loc[index]
                if(pred != y):
                    w += lr*y*x
                    if evaluate:
                        test_y_hat = self.__predict(self.test_x, w)
                        print(self.__calc_acc(self.test_y, test_y_hat))

        return w

    def __predict(self, X, w):
        # todo: assertion for w.shape == (x_df.shape[1]+1, 1)
        ones_column = np.ones((X.shape[0], 1))
        X = np.hstack((ones_column, X))
        return np.sign(np.dot(X, w))

    def __calc_acc(self, y, y_hat):
        correct_preds = np.sum(y == y_hat)
        return (correct_preds / len(y)) * 100


def main():
    perceptron = Perceptron('Dataset.csv')
    weights = perceptron.train(epoch=1, evaluate=True)


if __name__ == '__main__':
    main()
