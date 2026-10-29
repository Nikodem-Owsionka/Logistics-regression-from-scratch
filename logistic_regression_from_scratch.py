import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


red_wine_data = pd.read_csv("winequality-red.csv")

red_wine_data['quality'] = (red_wine_data['quality'] > 5).astype(int)

x = red_wine_data.drop('quality' , axis=1).values
y = red_wine_data['quality'].values

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=123)

#-----------------------------------------------------------------------------------------------

def standard_scaler(x_train , x_test):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train_scaled = (x_train - mean) / std
    x_test_scaled = (x_test - mean) / std

    return x_train_scaled , x_test_scaled

#-----------------------------------------------------------------------------------------------

x_train , x_test = standard_scaler(x_train , x_test)

rows , cols = x_train.shape

rows_test , cols_test = x_test.shape

cost = []
iteration = []

#-----------------------------------------------------------------------------------------------

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

#-----------------------------------------------------------------------------------------------

def CostFunction(x, y, m, c):
    
    value = 0

    for i in range(rows):

        z = np.dot(m,x[i]) + c
        g = Sigmoid(z)

        value += ((-y[i]) * (np.log(g))) - ((1-y[i]) * (np.log(1-g)))
    
    Cost = (1/rows) * value

    return Cost

#-----------------------------------------------------------------------------------------------

def GradientFunction(x, y, m, c):

    dC_dm = np.zeros(cols)
    dC_dc = 0

    for i in range(rows):

        z = np.dot(m , x[i]) + c
        g = Sigmoid(z)

        dC_dc += (g-y[i])

        for j in range(cols):

            dC_dm[j] += (g - y[i]) * x[i,j]
        
    dC_dm = (1/rows) * dC_dm
    dC_dc= (1/rows) * dC_dc

    return dC_dm , dC_dc

#-----------------------------------------------------------------------------------------------

def GradientDescent(x, y, iterations, a):
    m = np.zeros(cols)
    c = 0

    for n in range(iterations):
        
        dC_dm , dC_dc = GradientFunction(x, y, m, c)

        m = m - (a * dC_dm)
        c = c - (a * dC_dc)

        if n % 500 == 0:
            cost.append(CostFunction(x, y, m, c))
            iteration.append(n)

    return m , c

#-----------------------------------------------------------------------------------------------

def Predict(x, m, c):

    preds = np.zeros(rows_test)

    for i in range(rows_test):
        
        z = np.dot(m , x[i]) + c
        g = Sigmoid(z)

        if g >= 0.5:
            preds[i] = 1
        else:
            preds[i] = 0
    
    return preds

#-----------------------------------------------------------------------------------------------

Learning_rate = 0.01
Iterations = 20000

#-----------------------------------------------------------------------------------------------

final_m , final_c = GradientDescent(x_train , y_train , Iterations , Learning_rate)

predictions = Predict(x_test , final_m , final_c)

accuracy = np.mean(predictions == y_test) * 100

print(f"Model accuracy on data : {accuracy:.2f}%")

#-----------------------------------------------------------------------------------------------

def confusion_matrix(y_test, preds):
    tn = 0
    fn = 0
    tp = 0
    fp = 0

    for n in range(len(y_test)):

        if y_test[n] == 0 and preds[n] == 0:
            tn += 1
        elif  y_test[n] == 1 and preds[n] == 0:
            fn += 1
        elif y_test[n] == 1 and preds[n] == 1:
            tp += 1
        elif y_test[n] == 0 and preds[n] == 1:
            fp += 1

    return [[tn , fp],[fn , tp]]

#-----------------------------------------------------------------------------------------------

cm = confusion_matrix(y_test , predictions)

print(f"Confusion matrix : {cm}")

precision = ((cm[1][1])/(cm[1][1] + cm[0][1])) * 100

recall = ((cm[1][1])/(cm[1][1] + cm[1][0])) * 100

specificity = ((cm[0][0])/(cm[0][0] + cm[0][1])) * 100

print(f"Precision = {precision:.2f}%" )

print(f"Recall = {recall:.2f}%")

print(f"Specificity = {specificity:.2f}%")

#-----------------------------------------------------------------------------------------------

plt.xlabel("iteration")
plt.ylabel("cost")

plt.plot(iteration, cost, color="red")
plt.show()




