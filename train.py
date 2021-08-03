import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('tkagg')


filename = 'filename.csv'
names0 = ['sqft_lot','bedrooms','bathrooms','sqft_living']
samples = np.array(pd.read_csv(filename, names=names0, skiprows=[0], header=None)).astype(float)
where_are_NaNs = np.isnan(samples) # replaces NaNs with zeros
samples[where_are_NaNs] = float(0)
names1 = ['price']
labels =  np.array(pd.read_csv(filename, names=names1, skiprows=[0], header=None)).astype(float)
where_are_NaNs = np.isnan(labels)
labels[where_are_NaNs] = float(0)
y = labels
min_max_scaler = preprocessing.MinMaxScaler()


X = min_max_scaler.fit_transform(samples.astype(np.float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)



# fully/densely connected layers
model = Sequential([
    Dense(units=3, input_shape=(samples.shape[1],), activation='relu'),
    Dense(units=1, activation='relu'),])
model.summary()
num_epochs = 400


# training
model.compile(optimizer=Adam(learning_rate=0.1), loss='mse', metrics=['mae','mse'])
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=15,validation_data=(X_val, y_val), shuffle=True)


# PLOTTING:
loss_train = history.history['mae']
loss_val = history.history['val_mae']
rmse_final = np.sqrt(loss_train[-1])
print("Final Root Mean Squared Error = ", rmse_final)

epochs = np.arange(1,num_epochs+1)
plt.plot(epochs, loss_train, label='Training loss')
plt.plot(epochs, loss_val,label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss: Mean Squared Error [Dollars$^2$]')
plt.grid()
plt.legend()
plt.show()
