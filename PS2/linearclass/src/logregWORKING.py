import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    # *** START CODE HERE ***

    lr = LogisticRegression()
    lr.fit(np.log(x_train),y_train)
    lr_theta = lr.theta

    y_pred = lr.predict(x_eval)
    util.plot(x=np.log(x_eval), y=y_eval, theta=lr_theta,save_path='lr_image.png')
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1])

        count = 0
        diff = 0.1
        while diff > self.eps or count > self.max_iter:
            theta_prev = self.theta
            gz = 1/(1+np.exp(-theta_prev.dot(x.T)))
            G_J_theta = -np.mean((y-gz)*x.T,axis=1)

            H = np.zeros((x.shape[1],x.shape[1]))
            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    H[i][j] = np.mean(gz*(1-gz)*x[:,i]*x[:,j])
            
            self.theta = theta_prev - self.step_size*np.linalg.inv(H).dot(G_J_theta)
            diff = np.linalg.norm(self.theta - theta_prev, ord=1)
            count = count + 1

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        print(self.theta)
        y_pred = 1/(1+np.exp(-x.dot(self.theta)))
        return y_pred
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
