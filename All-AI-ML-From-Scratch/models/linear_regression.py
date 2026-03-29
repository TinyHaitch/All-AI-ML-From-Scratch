#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# import import_ipynb
from models.base_model import BaseModel


# In[2]:


class LinearRegression(BaseModel):
    '''
    Linear Regression model implementation from scratch.

    Inherits from BaseModel and provides concrete implementation
    fot fit() and predict() methods.
    '''

    def __init__(self, learning_rate=0.01, iterations=1000, lambda_=0.0): 
        ''' 
        Initialize LinearRegression model parameters.

        parameters
        ----------
        learning_rate : float
            Step size for gradient desecnt updates.
        iterations : int
            Number of iterations for gradient descent.
        lambda_ : float 
            My Regularization parameter (L2), default 0.0 (no regularization)

        Atributes
        ---------
        weights : ndarray of shape (n_features,)
            Model weights for each input feature. 
        bias: float
            Model bias term.
        '''
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None 
        self.cost_history = []
 
    def cost_function(self, X, y):
        ''' 
        Compute the cost for all examples.

        Parameters
        ----------
        X : ndarray shape (m, n)
            where m, number of example and n, number of features
        y : ndarray shape (m,)
            Target Value

        Return
        ------
        Cost: 
        '''

        ## Number of examples m, and number of features n,
        m, n = X.shape

        cost = 0.  
        for i in range(m):
            ## Making the prediction 
            pred = np.dot(X[i], self.weights) + self.bias
            square_error = (pred - y[i]) ** 2
            cost += square_error

        cost = cost / (2 * m) 


        ## L2 regularization 
        regularized_cost = 0
        for j in range(n):
            regularized_cost += self.weights[j] ** 2
        regularized_cost = (self.lambda_ / (2 * m)) * regularized_cost

        ## Regularized cost + cost 
        t_cost = regularized_cost + cost

        return t_cost

    def gradient_function(self, X, y): 
        ''' 
        Computes the gradient for Linear regression 
 
        Parameters
        ----------
          X : ndarray Shape (m,n)
              m, examples by n, features
          y : ndarray Shape (m,)
              Target value 
              
        Returns
        -------
          dj_dw : ndarray Shape (n,)
              The gradient of the cost w.r.t. the parameters w. 
          dj_db : (scalar)             
              The gradient of the cost w.r.t. the parameter b. 
        '''
        
        ## Number of examples m, and number of features n,
        m, n = X.shape
        
        ## Initializing gradient parameters
        dj_dw = np.zeros(n)
        dj_db = 0

        ## Compute Gradient
        for i in range(m):
            ## making the prediction
            pred = np.dot(X[i], self.weights) + self.bias
            error = pred - y[i]
            for j in range(n):
                dj_dw[j] += error * X[i, j]
            dj_db += error

        dj_dw = dj_dw / m 
        dj_db = dj_db / m 

        ## Regularization L2
        for j in range(n):
            dj_dw[j] += (self.lambda_ / m) * self.weights[j]

        return dj_dw, dj_db
            

    def fit(self, X, y):
        ''' 
        Train LinearRegression using Batch Gradient Descent 
        (Uses the entire dataset at each iteration to compute the gradient and update parameters)

        Parameters
        ----------
        X: ndarray of shape (m, n)
            Training data where m is the number of examples and n is the number of features

        y : ndarray of shape (m,)
            Target values corresponding to each training example.

        Return
        ------
        None
        
        Notes
        -----
        This implementation follows the classical batch gradient descent procedure:
        1. Compute predictions for each example
        2. Compute prediction error
        3. Accumulate gradients for weights and bias
        4. Average gradients
        5. Apply regularization (if lambda_ > 0)
        6. Update parameters
        '''

        ## Number of examples m, and number of features n,
        m, n = X.shape
        
        ## Initializing the parameters 
        self.weights = np.zeros(n)
        self.bias = 0

        for iter_ in range(self.iterations):

            dj_dw, dj_db = self.gradient_function(X, y)
            
            ## Gradient descent 
            self.weights = self.weights - self.learning_rate * dj_dw
            self.bias = self.bias - self.learning_rate * dj_db
            
            cost = self.cost_function(X, y)
            self.cost_history.append(cost)

            if iter_ % 100 == 0:
                print(f"Iteration {iter_}, Cost: {cost}")


    def predict(self, X): 
        ''' 
        Use the learned model to make predictions on new and unseen data 

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Input data where m is the number of examples and n is the number of features

        Return
        ------
        y_pred : ndarray of shape (m,) 
            Predicted target values for each input example.
            
        '''

        ## Check if model has been trained 
        if self.weights is None or self.bias is None:
            raise Exception('Model is not yet trained. Call the fit() before prediction(). ')

        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
                    


# In[ ]:


# In[3]:


# get_ipython().system('jupyter nbconvert --to script "Linear_Regression.ipynb" --output \'linear_regression\' --output-dir=\'../models/\'')


# In[ ]:


# In[ ]:




