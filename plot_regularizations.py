# -*- coding: utf-8 -*-

"""
=========================
Train error vs Test error
=========================
"""
print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import numpy as np
from sklearn.linear_model import Ridge

def plot_regularizations(X_train, X_test, y_train, y_test, alphas):
    # Compute train and test errors
    regressor = Ridge() # regularized linear regression
    train_errors = list()
    test_errors = list()
    for alpha in alphas:
        print("Woring on alpha = %s..." % alpha)
        regressor.set_params(alpha=alpha)
        regressor.fit(X_train, y_train)
        
        train_error = regressor.score(X_train, y_train)
        print("train_error : %s..." % train_error)
        train_errors.append(train_error)
        
        test_error = regressor.score(X_test, y_test)
        print("test_error : %s..." % test_error)
        test_errors.append(test_error)
    
    i_alpha_optim = np.argmax(test_errors)
    alpha_optim = alphas[i_alpha_optim]
    print("Optimal regularization parameter : %s" % alpha_optim)
    
    # Estimate the coef_ on full data with optimal regularization parameter
    regressor.set_params(alpha=alpha_optim)
    coef_ = regressor.fit(X_train, y_train).coef_
    
    # #############################################################################
    # Plot results functions
    
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.semilogx(alphas, train_errors, label='Train')
    plt.semilogx(alphas, test_errors, label='Test')
    plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
               linewidth=3, label='Optimum on test')
    plt.legend(loc='lower left')
    plt.ylim([0, 1.2])
    plt.xlabel('Regularization parameter')
    plt.ylabel('Performance')
    
    # Show estimated coef_ vs true coef
    plt.subplot(2, 1, 2)
    plt.plot(coef_, label='Estimated coef')
    plt.legend()
    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)
    plt.show()