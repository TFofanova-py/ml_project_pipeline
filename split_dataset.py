import numpy as np
from sklearn.model_selection import train_test_split
import pickle


X = np.load("datasets/dataset_fixed/X.npy")
Y = np.load("datasets/dataset_fixed/Y.npy")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, train_size=0.8)

with open("datasets/x_train.pickle", "wb") as f:
	pickle.dump(X_train, f)
with open("datasets/x_test.pickle", "wb") as f:
	pickle.dump(X_test, f)
with open("datasets/y_train.pickle", "wb") as f:
	pickle.dump(Y_train, f)
with open("datasets/y_test.pickle", "wb") as f:
	pickle.dump(Y_test, f)
	
