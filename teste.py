import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error,explained_variance_score
import numpy as np
import tensorflow as tf

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

def criar_modelo(input_dim, output_dim):
    model = Sequential()
    model.add(InputLayer(shape=(input_dim,)))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(144, activation='relu'))
    model.add(Dense(144, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
    return model

df = pd.read_csv('data/dataset/dados_treino_V3.csv')

X = df.drop(columns=['Município', 'Média_permanência', 'Valor_médio_intern', 'Taxa_mortalidade', 'Valor_total', 'Internações'])
y = df['Valor_médio_intern']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

scaler_X= StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()


input_dim = X_train_scaled.shape[1]
output_dim = 1
model = criar_modelo(input_dim, output_dim)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stop]
)
y_pred_Scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_Scaled.reshape(-1, 1)).flatten()  
mape = mean_absolute_percentage_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
variance_score = explained_variance_score(y_test, y_pred)

print(f"MAPE: {mape*100:.3f}%")
print(f"MAE: {mae:.3f}")
print(f"Variance Score: {variance_score:.3f}")