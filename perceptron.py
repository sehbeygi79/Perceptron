import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, file_path, train_data_ratio, seed=None):
        df = self.__do_preprocessings(pd.read_csv(file_path))

        self.train_df = df.sample(frac=train_data_ratio, random_state=seed)
        test_df = df.drop(self.train_df.index)
        self.test_x = test_df.loc[:, test_df.columns != 'Outcome'].values
        self.test_y = test_df['Outcome'].values.reshape(-1, 1)

    def __do_preprocessings(self, df):
        # remove corrupted data
        df = df.loc[(df['BMI'] != 0) & (df['Glucose'] != 0)
                    & (df['BloodPressure'] != 0)]
        x_df = df.loc[:, df.columns != 'Outcome']
        y_df = df.loc[:, df.columns == 'Outcome'].replace(
            to_replace={True: 1, False: -1})
        # normalize
        x_df = (x_df-x_df.min())/(x_df.max()-x_df.min())

        return pd.concat([x_df, y_df], axis=1, join='inner')

    # learning rate has no effect on perceptron but I used it anyway :)
    def train(self, target_acc, lr=1, epoch=1, evaluate=False):
        x_df = self.train_df.loc[:, self.train_df.columns != 'Outcome']
        y_df = self.train_df['Outcome']

        d = x_df.shape[1]
        w = np.zeros((d+1, 1))
        best_w, best_acc = w, 0
        iter_count = 0
        target_reached = False
        acc_list = []

        test_y_hat = self.__predict(self.test_x, w)
        curr_acc = self.__calc_acc(self.test_y, test_y_hat)
        acc_list.append(curr_acc)

        for i in range(epoch):
            for index, row in x_df.iterrows():
                x = np.insert(np.array(row), 0, 1).reshape(-1, 1)
                pred = np.sign(np.dot(x.T, w))
                y = y_df.loc[index]
                if(pred != y):
                    w += lr*y*x
                    # stop counting when target accuracy is reached
                    iter_count += 0 if target_reached else 1
                    if evaluate:
                        test_y_hat = self.__predict(self.test_x, w)
                        curr_acc = self.__calc_acc(self.test_y, test_y_hat)
                        acc_list.append(curr_acc)
                        if curr_acc > best_acc:
                            best_acc = curr_acc
                            best_w = w
                        if curr_acc >= target_acc:
                            target_reached = True

        return (best_w, best_acc, iter_count, acc_list)

    def __predict(self, X, w):
        ones_column = np.ones((X.shape[0], 1))
        X = np.hstack((ones_column, X))
        return np.sign(np.dot(X, w))

    def __calc_acc(self, y, y_hat):
        correct_preds = np.sum(y == y_hat)
        return correct_preds / len(y)

def main():
    # seed is a parameter for df.sample method which separates train/test data
    # you can set a seed to get repeatable results 
    perceptron = Perceptron('Dataset.csv', train_data_ratio=0.85, seed=None)
    best_w, best_acc, iter_count, acc_list = perceptron.train(
        target_acc=0.7, epoch=1, evaluate=True)

    print(
        f'number of iterations needed to reach minimum of 70% accuracy on test data: {iter_count}')
    print(f'best reached accuracy on test data: {best_acc}')
    plt.plot(np.array(acc_list))
    plt.show()


if __name__ == '__main__':
    main()
