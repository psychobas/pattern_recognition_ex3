import numpy as np


class LOGREG(object):
    '''
    Logistic regression class based on the LOGREG lecture slides
    '''

    def __init__(self, regularization: float = 0):
        self.r = regularization
        self._threshold = 10e-9
        self._eps = self._threshold

    def activationFunction(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        # TODO: Implement logistic function -> DONE
        #print("shape of w is: ", w.shape)
        #print("shape of X is: ", X.shape)
        z = np.dot(w.T, X)
        #print("z is: ", z)
        #print("z shape is: ", z.shape)
        return 1 / (1 + np.exp(-z))

    def _costFunction(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Compute the cost function for the current model parameters
        :param w: current model parameters
        :param X: data
        :param y: data labels
        :return: cost
        '''

        m = len(y)
        h = self.activationFunction(w, X)
        #print("h is: ", h)
        #cost = ((((y).T @ np.log(h + self._eps)) - ((1 - h).T @ np.log(1 - h + self._eps))) / m).mean()


        #https://stackoverflow.com/questions/58567344/python-logistic-regression-hessian-getting-a-divide-by-zero-error-and-a-singu
        cost = (-sum(-y * np.log(h + self._eps) - (1 - y) * np.log(1 - h)) / m).flat[0]
        #print("cost is:   ", cost)

        regularizationTerm = - self.r / 2 * np.dot(w.T, w)
        #regularizationTerm = 0
        # TODO: Implement equation of cost function for posterior p(y=1|X,w) -> DONE, (check regularization)



        return cost + regularizationTerm

    def _calculateDerivative(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Compute the derivative of the model parameters
        :param w: current model parameters
        :param X: data
        :param y: data labels
        :return: first derivative of the model parameters
        '''
        # TODO: Calculate derivative of loglikelihood function for posterior p(y=1|X,w) -> Done

        h = self.activationFunction(w, X)

        #print("shape of X is: ", X.shape)
        #print("shape of h is: ", h.shape)
        #print("shape of y is: ", y.shape)
        #print("shape of reshaped x is: ", X.shape)

        firstDerivative = (np.dot(X, (h.flatten() - y.flatten())) / y.shape[0])
        #print("first Derivative is: ", firstDerivative)
        #double check!
        regularizationTerm = self.r * w
        regularizationTerm = 0

        return firstDerivative + regularizationTerm

    def _calculateHessian(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        '''
        :param w: current model parameters
        :param X: data
        :return: the hessian matrix (second derivative of the model parameters)
        '''
        # TODO: Calculate Hessian matrix of loglikelihood function for posterior p(y=1|X,w)

        #https://github.com/DrIanGregory/MachineLearning-LogisticRegressionWithGradientDescentOrNewton/blob/master/logisticRegression.py

        h = self.activationFunction(w, X)
        #
        #W = np.diag(h * (1 - h))
        print("h is: ", h)



        #hessian = X.T.dot(w).dot(X);
        #https://stackoverflow.com/questions/58567344/python-logistic-regression-hessian-getting-a-divide-by-zero-error-and-a-singu check!
        hessian = (np.dot(X, X.T) * np.diag(h) * np.diag(1 - h))
        print("hessian is: ", hessian)
        #print("shape of X is: ", X.shape)
        #print("shape of h is: ", h.shape)
        #hessian = X.dot(h.T)#.dot(X)
        print("hessian raw shape is", hessian.shape)
        regularizationTerm = self.r
        return - hessian + regularizationTerm

    def _optimizeNewtonRaphson(self, X: np.ndarray, y: np.ndarray, number_of_iterations: int) -> np.ndarray:
        '''
        Newton Raphson method to iteratively find the optimal model parameters (w)
        :param X: data
        :param y: data labels (0 or 1)
        :param number_of_iterations: number of iterations to take
        :return: model parameters (w)
        '''
        # TODO: Implement Iterative Reweighted Least Squares algorithm for optimization, use the calculateDerivative and calculateHessian functions you have already defined above
        w = np.zeros((X.shape[0], 1))

        posteriorloglikelihood = self._costFunction(w, X, y)
        print('initial posteriorloglikelihood', posteriorloglikelihood, 'initial likelihood',
              np.exp(posteriorloglikelihood))

        for i in range(number_of_iterations):
            print("i is: ", i)
            oldposteriorloglikelihood = posteriorloglikelihood
            print("posteriorloglik is: ", np.exp(posteriorloglikelihood))
            w_old = w
            h = self._calculateHessian(w, X)
            #print("shape of hessian is: ", h.shape)
            #print("shape of w_old is: ", w_old.shape)
            derivative = self._calculateDerivative(w, X, y)
            #print("shape of derivative is: ", derivative.shape)
            #slide 21
            w_update = np.dot(np.linalg.inv(h), derivative)
            print("w_update is: ", w_update)
            #print("shape of w_update is: ", w_update.shape)
            w = w_old.T + w_update
            w = w.reshape(-1,1)
            posteriorloglikelihood = self._costFunction(w, X, y)
            #print("posteriorloglikelihood is: ", posteriorloglikelihood)
            if self.r == 0:
                # TODO: What happens if this condition is removed? -> we get a singular matrix error when computing the dot product for the w_update
                if np.exp(posteriorloglikelihood) > 1 - self._eps:
                    print('posterior > 1-eps, breaking optimization at niter = ', i)
                    break

            # TODO: Implement convergence check based on when w_update is close to zero
            # Note: You can make use of the class threshold value self._threshold
            if np.any(abs(w_update) < self._threshold):
                print("min of w_update is: ", min(w_update))
                print(np.any(abs(w_update) < self._threshold))
            #if w_update.mean() < self._threshold:
                break

        print('final posteriorloglikelihood', posteriorloglikelihood, 'final likelihood',
              np.exp(posteriorloglikelihood))

        # Note: maximize likelihood (should become larger and closer to 1), maximize loglikelihood( should get less negative and closer to zero)
        return w

    def train(self, X: np.ndarray, y: np.ndarray, iterations: int) -> np.ndarray:
        '''
        :param X: dataset
        :param y: ground truth labels
        :param iterations: Number of iterations to train
        :return: trained w parameter
        '''
        self.w = self._optimizeNewtonRaphson(X, y, iterations)
        return self.w

    def classify(self, X: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained logistic regressor - access the w parameter through self.
        :param x: Data to be classified
        :return: List of classification values (0.0 or 1.0)
        '''
        # TODO: Implement classification function for each entry in the data matrix
        numberOfSamples = X.shape[1]

        probabilities = self.activationFunction(w = self.w, X = X)
        predictions = np.where(probabilities < 0.5, 0, 1)
        #print("predictions is: ", predictions)

        #predictions = ???
        return predictions


    def printClassification(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls "classify" and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''

        #added by me
        predictions = self.classify(X)

        pred_minus_truth = predictions - y

        numOfMissclassified = np.count_nonzero(pred_minus_truth)


        totalError = numOfMissclassified / X.shape[1]

        #print("predictions is: ", predictions)

        # TODO: Implement print classification
        numberOfSamples = X.shape[1]

        print("{}/{} misclassified. Total error: {:.2f}%.".format(numOfMissclassified, numberOfSamples, totalError))
