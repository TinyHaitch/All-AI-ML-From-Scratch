#!/usr/bin/env python
# coding: utf-8

# ## This module is for the BaseModel 
# - The BaseModel will serve as skeleton for my learning algorithms
# - It exists to standardize my learning algorithms
# - This ensure all learning models has the same interface .fit() and .predict() method
# - This protects my models to have different method names

# In[3]:


class BaseModel: 
    '''
    Base class for all learning algorithms in this project.

    This class defines the common interface that every model must follow.
    Any model that inherits from BaseModel must implement the fit() and the predict() methods in them.
    '''

    def fit(self, X, y): 
        '''
        Train the model using provided dataset.

        Parameters 
        -----------
        X : ndarray of shape (m, n)
            Training data where m is the number of examples and n is the number of features.

        y : ndarray of shape (m,)
             Target values corresponding to each training example.

        Returns
        -------
        None 

        Notes
        -----
        This method must be implemented in subclasses 
        '''
        raise NotImplementedError('Subclasses must implement fit()')


    def predict(self, X): 
        ''' 
        Generate predictions using the trained model from the fit() method

        parameters
        ----------
        X : ndarray of shape(m, n)
            Input data for which predictions should be made 

        Returns
        -------
        y_pred : ndarray of shape(m,)
            Predicted values for each example.

        Notes
        -----
        This method must be implemented in subclasses
        '''
        raise NotImplementedError('Subclasses must implement predict()')


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script base_model.ipynb')


# In[ ]:




