import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # We can also import naive_bayes or svc from sklearn models libraru
from sklearn import metrics

#1 Load in the dataset
data_frame = pd.read_csv("./smsspamcollection.tsv", sep="\t")

#2 Split the data into training and testing sets
X = data_frame[['length','punct']]
y = data_frame['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#3 Create and Train a model
lr_model = LogisticRegression(solver="lbfgs")
lr_model.fit(X_train,y_train)

#Evaluate the model
predictions = lr_model.predict(X_test)
predictions_result = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index = ["ham","spam"], columns = ["ham","spam"])
print(predictions_result)

print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))