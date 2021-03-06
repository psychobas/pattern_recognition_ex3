import numpy as np


class LOGREG(object):
    '''
    Logistic regression class based on the LOGREG lecture slides
    '''


    def __init__(self, regularization: float = 0):
        self.r = regularization
        self._threshold = 10e-6
        self._eps = self._threshold

    def activationFunction(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        # TODO: Implement logistic function -> DONE, CHECKED
        # print("shape of w is: ", w.shape)
        # print("shape of X is: ", X.shape)
        z = np.dot(w.T, X)
        # print("z is: ", z)
        # print("z shape is: ", z.shape)
        result = 1.0 / (1 + np.exp(-z))
        return result

    def _costFunction(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Compute the cost function for the current model parameters
        :param w: current model parameters
        :param X: data
        :param y: data labels
        :return: cost
        '''

        m = len(y)
        # h = self.activationFunction(w, X) # seems unecessary, doing more than you should
        h = np.dot(w.T, X)  # h is shorthand for w^T*x

        # print("h is: ", h)
        # cost = ((((y).T @ np.log(h + self._eps)) - ((1 - h).T @ np.log(1 - h + self._eps))) / m).mean()

        # https://stackoverflow.com/questions/58567344/python-logistic-regression-hessian-getting-a-divide-by-zero-error-and-a-singu
        # cost = (-sum(-y * np.log(h + self._eps) - (1 - y) * np.log(1 - h)) / m).flat[0]
        # print("cost is:   ", cost)

        # slide 17!
        cost = np.sum(y * h - np.log(1 + np.exp(h))) / m
        # cost = (-y * np.log(h + self._eps) - (1 - y) * np.log(1 - h)).mean()
        # cost = y * w.T * X - np.log()


        # print("shape of w is: ", w.shape)
        #print("shape of w is: ", w.shape)
        # make sure bias term is not regularized
        regularizationTerm =  -self.r / 2 * np.sum(np.square(w))   #np.diag(- self.r / 2 * np.dot(w.T, w))
        #print("regularization Term is: ", regularizationTerm)

        # is r = 1/(sigma^2)?
        # print("Regularization term is: ", regularizationTerm)
        # regularizationTerm = 0
        # TODO: Implement equation of cost function for posterior p(y=1|X,w) -> DONE, CHECKED (questions remain)
        #print("cost is: ", cost)
        #print("exp of cost is: ", np.exp(cost))

        return cost + regularizationTerm

    def _calculateDerivative(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Compute the derivative of the model parameters
        :param w: current model parameters
        :param X: data
        :param y: data labels
        :return: first derivative of the model parameters
        '''
        # TODO: Calculate derivative of loglikelihood function for posterior p(y=1|X,w) -> DONE, CHECKED
        DEBUG = False
        h = self.activationFunction(w, X)
        numOfEntries = X.shape[1]
        dim = X.shape[0]
        #print("shape of X is: ", X.shape)
        #print("shape of h - y is: ", (h-y).shape)
        # print("shape of y is: ", y.shape)
        # print("shape of reshaped x is: ", X.shape)

        #murphy page 247
        #firstDerivative = (np.dot(X, (h.flatten() - y.flatten())) / y.shape[0]).T # I do not know what formula you are
                                                                                # using, but take a look at slide 20

        firstDerivative = np.dot(X, (h - y).T)
        # my_first_derivative = np.dot(X, (h - y).T) / numOfEntries
        #
        # print("my first derivative is:", my_first_derivative)
        # firstDerivative = np.zeros((dim, 1))
        # print("shape of first derivative is: ", firstDerivative.shape)
        # for i in range(numOfEntries - 1):
        #     x_i = X[:, i]
        #     y_i = y[i]
        #     h_i = h[:,i]
        #     if DEBUG:
        #         print(np.shape(X))
        #         print(np.shape(y))
        #         print(np.shape(h))
        #         print(np.shape(x_i))
        #         print(np.shape(y_i))
        #         print(np.shape(h_i))
        #         print(np.shape(y_i-h_i))
        #         print(np.shape((y_i-h_i)*x_i.T))
        #         #print(np.shape(firstDerivative[:,i]))
        #         print(np.shape(firstDerivative[1,:]))
        #         print("first derivative is: ", firstDerivative)
        #     additionTerm = (y_i-h_i)*x_i.T
        #     additionTerm = np.reshape(additionTerm, (3,1))
        #     firstDerivative += additionTerm
        #
        #     firstDerivative = firstDerivative / numOfEntries
        #
        #     #the two methods for calculating the derivatives yield different results
        #     print("derivatives are the same:", my_first_derivative == firstDerivative)


        # firstDerivative = np.sum((y - h) * X.T)  # not working

        # print("h flattened is: ", h.flatten())
        # print("y flattened is: ", y.flatten())
        # print("first Derivative is: ", firstDerivative)
        # double check!
        # regularizationTerm = self.r * w.T
        #regularizationTerm = np.diag(- self.r * w.T)  # I hope that r is = 1/sigma^2, slide 27, is whole w needed or w[1:]?
        regularizationTerm = 2 * self.r * w.T
        #print("regularization term is:", regularizationTerm)
        regularizationTerm[0][0] = 0

        #print("regularization term: ", regularizationTerm)
        #print("reg_term: ", regularizationTerm)
        # regularizationTerm = 0

        # print("first Derivative is: ", firstDerivative)


        return firstDerivative + regularizationTerm.T

    def _calculateHessian(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        '''
        :param w: current model parameters
        :param X: data
        :return: the hessian matrix (second derivative of the model parameters)
        '''
        # TODO: Calculate Hessian matrix of loglikelihood function for posterior p(y=1|X,w) -> DONE, CHECKED

        # https://github.com/DrIanGregory/MachineLearning-LogisticRegressionWithGradientDescentOrNewton/blob/master/logisticRegression.py

        h = self.activationFunction(w, X)
        #print("H SHAPE IS",h.shape)
        #
        # W = np.diag(h * (1 - h))
        # print("h is: ", h)

        # hessian = X.T.dot(w).dot(X);
        # https://stackoverflow.com/questions/58567344/python-logistic-regression-hessian-getting-a-divide-by-zero-error-and-a-singu check!
        #hessian = -(np.dot(X, X.T) * np.diag(h) * np.diag(1 - h)) # not sure why you are using diagonals here
        #hessian = -(np.dot(X, X.T) * np.diag(h * (1-h))) * np.dot(X, X.T) /(X.shape[0]) best result so far with this

        #print("h-h", np.diagflat(h * (1-h)))


        hessian = -np.dot((np.dot(X, np.diagflat(h * (1-h)))), X.T)
        #print("shape 1 of hessian is: ", hessian.shape)

        #print("hessian is: ", hessian.shape)

        #hessian = - np.sum((np.dot(X, X.T) * (h * (1 - h)))  # slide 22, does not work due to mismatched dimensions
                                                              # trusting your formula instead

        #print("shape of hessian is: ", hessian.shape)
        # print("hessian is: ", hessian)
        # print("shape of X is: ", X.shape)
        # print("shape of h is: ", h.shape)
        # hessian = X.dot(h.T)#.dot(X)
        # print("hessian raw shape is", hessian.shape)
        regularizationTerm = - self.r  # does the transpose affect the sign? or set the term to 0?
        reg_matrix = np.zeros((hessian.shape[0], hessian.shape[0]))
        np.fill_diagonal(reg_matrix, regularizationTerm)

        #print("hessian is: ", hessian)
        #print("hessian diff :", hessian - (hessian + reg_matrix))
        return hessian + reg_matrix

    def _optimizeNewtonRaphson(self, X: np.ndarray, y: np.ndarray, number_of_iterations: int) -> np.ndarray:
        '''
        Newton Raphson method to iteratively find the optimal model parameters (w)
        :param X: data
        :param y: data labels (0 or 1)
        :param number_of_iterations: number of iterations to take
        :return: model parameters (w)
        '''
        # TODO: Implement Iterative Reweighted Least Squares algorithm for optimization
        #  use the calculateDerivative and calculateHessian functions you have already defined above -> DONE, CHECKED
        w = np.zeros((X.shape[0], 1))

        posteriorloglikelihood = self._costFunction(w, X, y)
        # print('initial posteriorloglikelihood', posteriorloglikelihood, 'initial likelihood',
        #      np.exp(posteriorloglikelihood))

        for i in range(number_of_iterations):
            # print("i is: ", i)
            oldposteriorloglikelihood = posteriorloglikelihood
            # print("posteriorloglik is: ", np.exp(posteriorloglikelihood))
            w_old = w
            h = self._calculateHessian(w, X)
            # print("shape of hessian is: ", h.shape)
            # print("shape of w_old is: ", w_old.shape)
            derivative = self._calculateDerivative(w, X, y)
            # print("shape of derivative is: ", derivative.shape)
            # slide 21
            # slide 23 bottom: Matrix formula!!
            inverse = np.linalg.inv(h)
            #print("shape of inverse is: ", inverse.shape)
            #print("shape of derivative is: : ", derivative.shape)
            w_update = -np.dot(inverse, derivative)
            #print("w_update is: ", w_update)
            #print("w_update is: ", w_update)
            # print("shape of w_update is: ", w_update.shape)
            w = w_old - w_update
            # w = w.reshape(-1,1)  #unnecessary (the way I see it)
            posteriorloglikelihood = self._costFunction(w, X, y)
            # print("posteriorloglikelihood is: ", posteriorloglikelihood)

            if self.r == 0:
                # TODO: What happens if this condition is removed?
                #  -> we get a singular matrix error when computing the dot product for the w_update DONE, CHECKED
                if np.any(np.exp(posteriorloglikelihood) > 1 - self._eps):
                    print('posterior > 1-eps, breaking optimization at niter = ', i)
                    break

            # TODO: Implement convergence check based on when w_update is close to zero -> DONE, CHECKED
            # Note: You can make use of the class threshold value self._threshold
            if np.any(abs(w_update) < self._threshold):
                # print("min of w_update is: ", min(w_update))
                # print(np.any(abs(w_update) < self._threshold))
                # if w_update.mean() < self._threshold:
                print("breaking because np.any(abs(w_update)) < self._threshold")
                break

        # print('final posteriorloglikelihood', posteriorloglikelihood, 'final likelihood',
        #      np.exp(posteriorloglikelihood))

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
        print("number of iteratinos is: ", iterations)
        return self.w

    def classify(self, X: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained logistic regressor - access the w parameter through self.
        :param x: Data to be classified
        :return: List of classification values (0.0 or 1.0)
        '''
        # TODO: Implement classification function for each entry in the data matrix -> DONE, CHECKED
        numberOfSamples = X.shape[1]

        probabilities = self.activationFunction(self.w, X)
        #print("probabilities are: ", probabilities)
        predictions = np.where(probabilities < 0.5, 0, 1)
        # print("predictions is: ", predictions)

        # predictions = ???
        return predictions

    def printClassification(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls "classify" and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''

        # added by me
        predictions = self.classify(X)



        pred_minus_truth = predictions - y

        numOfMissclassified = np.count_nonzero(pred_minus_truth)
        numberOfSamples = X.shape[1]
        totalError = (numOfMissclassified / numberOfSamples) * 100

        # print("predictions is: ", predictions)

        # TODO: Implement print classification -> DONE, CHECKED


        print("{}/{} misclassified. Total error: {:.2f}%.".format(numOfMissclassified, numberOfSamples, totalError))
