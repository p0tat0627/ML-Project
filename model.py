import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('features.csv')

X = df.iloc[: ,:-1].values
Y = df['labels'].values

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)



# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y.reshape(-1))

# scaling our data with sklearn's Standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(x_train,y_train)

pickle.dump(sv, open('features.pkl', 'wb'))