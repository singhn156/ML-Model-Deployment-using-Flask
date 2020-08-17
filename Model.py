# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import pickle


dataset = pd.read_csv('student_scores - student_scores.csv')

dataset.head()
# Getting summary of dataset
dataset.describe()

# Check linearity assumption
plt.plot(dataset['Hours'],dataset["Scores"],'o',color="black")
plt.title("Linear assumption Exists")

# Check Normality assumption
stats.probplot(dataset['Hours'],dist='norm',plot=plt)
plt.show()

#Check Correlation assumption
dataset.corr()

X=dataset.iloc[:,0].values
y= dataset.iloc[:,-1].values

X=X.reshape(-1, 1)
y=y.reshape(-1, 1)
#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[9.25]]))