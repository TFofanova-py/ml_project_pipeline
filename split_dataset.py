import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import yaml


X = np.load("datasets/dataset_fixed/X.npy")
Y = np.load("datasets/dataset_fixed/Y.npy")

with open("params.yaml", "r") as f:
	params = yaml.safe_load(f)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, train_size=params["split_dataset"]["train_size"])

with open("datasets/x_train.pickle", "wb") as f:
	pickle.dump(X_train, f)
with open("datasets/x_test.pickle", "wb") as f:
	pickle.dump(X_test, f)
with open("datasets/y_train.pickle", "wb") as f:
	pickle.dump(Y_train, f)
with open("datasets/y_test.pickle", "wb") as f:
	pickle.dump(Y_test, f)
	
