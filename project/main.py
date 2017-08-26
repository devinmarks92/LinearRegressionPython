import numpy as np
import linear_regression as lr
import extract_data as extract

data_filepath = input('Enter file path for data set > ')
param_filepath = input('Enter file path with prediction parameters > ')

m, y, X = extract.datafile_values(data_filepath)
parameters = extract.paramfile_values(param_filepath)
theta = np.zeros((m, 1))
X_norm, param_norm = lr.normalize(X, parameters, m)

# Normal equation
print('\nAttemping normal equation...')
if m < 100000:
    theta = lr.normal_equation(X, y)
else:
    print('Number of parameters too large')

print('Prediction via normal equation: ')
print(str(float(np.dot(theta.T, parameters.T))) + '\n')

#Linear Regression
print('Attemping linear regression...')
print('Suggested alpha: 0.01; Suggested iterations: 400')
alpha = float(input('Enter alpha value > '))
num_iter = int(input('Enter number of iterations > '))
theta, J_history = lr.gradient_descent(X_norm, y, m, alpha, num_iter)

print('Prediction via linear regression: ')
print(str(float(np.dot(theta.T, param_norm.T))) + '\n')
lr.plot_descent(range(num_iter), J_history)
