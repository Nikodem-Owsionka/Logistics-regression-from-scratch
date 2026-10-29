import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


red_wine_data = pd.read_csv("winequality-red.csv")

red_wine_data['quality'] = (red_wine_data['quality'] > 5).astype(int)

x = np.array(red_wine_data.drop('quality' , axis=1))
y = np.array(red_wine_data['quality'])

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=123)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()

model.fit(x_train , y_train)

y_preds = model.predict(x_test)

accuracy = accuracy_score(y_test , y_preds) * 100

print(f"model accuracy on data: {accuracy:.2f}")

cm = confusion_matrix(y_test , y_preds)

print(f"Confusion matrix : {cm}")

precision = ((cm[1][1])/(cm[1][1] + cm[0][1])) * 100
recall = ((cm[1][1])/(cm[1][1] + cm[1][0])) * 100
specificity = ((cm[0][0])/(cm[0][0] + cm[0][1])) * 100

print(f"Precision = {precision:.2f}%" )
print(f"Recall : {recall:.2f}%")
print(f"Specificity = {specificity:.2f}%")




