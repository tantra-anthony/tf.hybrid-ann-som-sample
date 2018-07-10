# identifying the frauds using SOM

# refer to self_organising_maps_sample.py to get the code to find the SOM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(7, 5)], mappings[(2, 1)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# now we move towards the supervised region
# we need a dependent variable now
# create a matrix of features first
# isFraud 0 means no fraud 1 means got fraud
customers = dataset.iloc[:, 1:].values

# now we need to create the dependent variable
# what is this dependent variable? it will be a binary outcome
# 0 if no fraud 1 is fraud, we can extract from the previous SOM
# first we need to initialize a vector of 0s and extract customer ids from SOM
# then change the respective SOMs into 1
# use len(dataset)
is_fraud = np.zeros(len(dataset))

# put 1 for customers which potentially cheated
# then replcae by 1
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds: # i corresponds to the ith line, 0 is index 0 of the column
        is_fraud[i] = 1 # make it 1 if got match with the customer ids

# now we have all the things we need for an ANN
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# dataset is so simple that 2 epochs is sufficient to see correlation
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Predicting the Test set results using probability of fraud
# use our classifier on inputs
y_pred = classifier.predict(customers)

# now we need to rank them, sort an array
# in concatenate we still need to make it a 2D array, same as y_pred
# as we need to concatenate horizontally axis is 1
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
# now we have the customer id and the probabiltiy

# then we sort their probabiltiy of cheating
# specify the column that we're going to sort
y_pred = y_pred[y_pred[:, 1].argsort()]



