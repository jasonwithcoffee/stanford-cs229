import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        X_transpose = X.transpose()
        self.theta = np.linalg.solve(X_transpose.dot(X),X_transpose.dot(y))

        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        polyList = [*range(k+1)]
        col_1 = np.array([X[:,0],]).transpose()
        col_2 = np.array([X[:,1],]).transpose()

        clone_col_2 = np.tile(col_2, (1,k))
        feature_cloned = np.hstack([col_1,clone_col_2])
        feature_poly = np.power(feature_cloned,polyList)
        return feature_poly

        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        polyList = [*range(k+1)]
        col_1 = np.array([X[:,0],]).transpose()
        col_2 = np.array([X[:,1],]).transpose()

        clone_col_2 = np.tile(col_2, (1,k))
        feature_cloned = np.hstack([col_1,clone_col_2])
        feature_poly = np.power(feature_cloned,polyList)

        col_2_sin = np.sin(col_2)
        feature_poly_sin = np.hstack([feature_poly,col_2_sin])
        return feature_poly_sin
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        pred_y = X.dot(self.theta)
        return pred_y
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        lm = LinearModel()

        # Create poly or sine features using training data
        train_x_feat = lm.create_poly(k,train_x)
        if sine==True:
            train_x_feat = lm.create_sin(k,train_x)

        lm.fit(train_x_feat,train_y)

        # Transform test data
        plot_x_feat = lm.create_poly(k,plot_x)
        if sine==True:
            plot_x_feat = lm.create_sin(k,plot_x)

        plot_y = lm.predict(plot_x_feat)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    # plt.clf() # NEED TO UNCOMMENT THIS


def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    run_exp(train_path=train_path)
    run_exp(train_path=train_path,sine=True)

    run_exp(train_path=small_path)
    run_exp(train_path=small_path,sine=True)    
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
