import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)

    gda = GDA()
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(3)

        n, d = x.shape
        y0 = 1-y
        phi = y.mean()
        mu0 = np.mean(x[y == 0], axis=0).reshape(d, 1)
        mu1 = np.mean(x[y == 1], axis=0).reshape(d, 1)
        sigma = (x.T - np.tile(mu0.T,(n,1)).T).dot((x - np.tile(mu1.T,(n,1))))/n

        theta1 = np.matmul(np.linalg.pinv(sigma),(mu1-mu0))
        theta0 = 0.5*(np.matmul(mu0.T,np.matmul(np.linalg.pinv(sigma),mu0))\
            - np.matmul(mu1.T,np.matmul(np.linalg.pinv(sigma),mu1))) - np.log((1-phi)/phi)
        self.theta = np.append(theta0,theta1).reshape(3,1)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        z = np.matmul(x, self.theta)
        gz = 1/(1 + np.exp(-z))
        gz = np.squeeze(gz,1)
        return gz
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
