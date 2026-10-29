# Logistics regression: from scratch vs scikit-learn

## Overview:

This project explores the mathematical foundations of binary logistics regression by implementing the algorithm from scratch in Python and comparing the results with scikit-learn's implementation.

I will be using the 'winequality-red.csv' data set from Kaggle to predict the quality of red wine based on its physicochemical properties. Wines with a quality score of greater than 5 will be classified as good (1) and wines with a quality score of 5 or below will be classified as not good (0).

## Objectives:

- Understand the mathematical principles underlying logistics regression
- Implement logistics regression from scratch using python
- Compare the results of my custom model to scikit-learns built-in Logistics regression model

## Method:

- Start by creating a function to standardise the input features
- Then start with a linear combination of the data
- Create a sigmoid function which returns a value between 0 and 1 (this will be the probability)
- Create a function to calculate the cost
- Create a gradient function to compute the derivatives of the cost function
- Create a function that carries out the gradient descent algorithm
- Create a function to return the prediction for each training data point
- Create a function to find the confusion matrix
- Display the results
- Implement a second model using scikit-learn's LogisticRegression class

#### Standardisation:

this improves the convergence of gradient descent since all features will be on a similar scale. Scaling the data in this way causes it to have a mean of 0 and standard deviation of 1

#### Gradient descent algorithm:

- starting with initial values of _m_ and _c_
- repeatedly update _m_ and _c_ in small increments for a certain number of iterations to reduce the mean square error and converge to the minimum value
- this is done using a formula that includes the partial derivatives of the mean square error and a fixed value for the learning rate to control the step size

#### Mathematical formulas:

found in 'Logistics regression Maths.pdf'

## Libraries needed:

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Results and findings

#### The confusion matrix:

<img width="113" height="61" alt="image" src="https://github.com/user-attachments/assets/691bfeed-1228-4ab4-9a16-98cddeae2b8a" />

#

- I correctly classified 343 wines out of 480, giving an accuracy of 71.5%

- out of 271 wines that i predicted as good quality, i predicted 191 of them correctly, giving a precision of 70.5%

- out of 248 wines that were actually good quality, i predicted 191 of them correctly, giving a recall of 77.0%.

- out of 232 wins that were actually lower quality, i predicted 152 of them correctly, giving a specificity of 65.5%

#

The custom implementation produced the same reported metrics as the scikit-learn model in my comparison.

Therefore these results demonstrate that my implementation can reproduce the performance of an established machine learning library on this binary classification task.

#### Visualisation:

Convergence of the cost for my custom model:

<img width="623" height="477" alt="image" src="https://github.com/user-attachments/assets/83a79897-9245-4f79-92db-7059905bf080" />

## Future improvements

- Compare performance with other classification algorithms
- Investigate the effect of learning rate and number of iterations on convergence
  
