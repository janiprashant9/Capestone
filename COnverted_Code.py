# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# %%
df = pd.read_csv('ESH22-900-ESH22.csv')

# %%
print(df.head())

# %%
df.isnull().sum()

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['OPEN(H)', 'HIGH(H)', 'LOW(H)', 'CLOSE(H)']])

# %%
def create_sequences(data, look_back=1):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 1
X, y = create_sequences(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# %%
train_size = int(len(X) * 0.70)
test_size = len(X) - train_size
trainX, testX = X[0:train_size], X[train_size:len(X)]
trainY, testY = y[0:train_size], y[train_size:len(y)]

# %%
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# %%
model.fit(trainX, trainY, epochs=20, batch_size=32)

# %%
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# %%

# Create a temporary array with the same shape as the scaled_data used to fit the scaler
temp_array = np.zeros((len(trainPredict), 4))
temp_array[:,0] = trainPredict[:,0]  # Copy predictions into the first column

# Inverse transform the temporary array
trainPredict_inverse = scaler.inverse_transform(temp_array)[:,0]

# Repeat the process for testPredict
temp_array = np.zeros((len(testPredict), 4))
temp_array[:,0] = testPredict[:,0]
testPredict_inverse = scaler.inverse_transform(temp_array)[:,0]

# If trainY is 1-dimensional, reshape it to 2-dimensional
trainY = trainY.reshape(-1, 1) if trainY.ndim == 1 else trainY
temp_array = np.zeros((len(trainY), 4))
temp_array[:,0] = trainY[:,0]
trainY_inverse = scaler.inverse_transform(temp_array)[:,0]

# Ensure testY is 2-dimensional
testY = testY.reshape(-1, 1) if testY.ndim == 1 else testY
temp_array = np.zeros((len(testY), 4))
temp_array[:,0] = testY[:,0]
testY_inverse = scaler.inverse_transform(temp_array)[:,0]

# %%
def mean_percentage_accuracy(y_true, y_pred):
    """
    Calculate the mean percentage accuracy.
    
    Parameters:
    - y_true: actual values
    - y_pred: predicted values
    
    Returns:
    - accuracy: mean percentage accuracy
    """
   
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)
    
  
    percentage_diff = np.abs((y_true - y_pred) / y_true) * 100
    
   
    accuracy = 100 - np.mean(percentage_diff)
    
    return accuracy

accuracy = mean_percentage_accuracy(testY_inverse, testPredict_inverse)
#prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')
print('Accuracy: ',accuracy)

# %%

plt.figure(figsize=(10,6))
plt.plot(testY_inverse, color='blue', label='Actual')  # Use testY_inverse instead of y_test
plt.plot(testPredict_inverse, color='red', label='Predicted')  # Use testPredict_inverse instead of predictions
plt.title('Forecasting with LSTM')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# %%
prediction = pd.DataFrame(testPredict_inverse, columns=['Predictions']).to_csv('OCLH_Predictions1.csv')
actual = pd.DataFrame(testY_inverse, columns=['Actual']).to_csv('OCLH_actual.csv')


