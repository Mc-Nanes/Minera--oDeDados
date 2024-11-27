import streamlit as st
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, explained_variance_score
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Configuração de semente
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Função para criar o modelo
def criar_modelo(input_dim, output_dim):
    model = Sequential()
    model.add(InputLayer(shape=(input_dim,)))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(144, activation='relu'))
    model.add(Dense(144, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
    return model

# Interface do Streamlit
st.title("Treinamento de Rede Neural com Streamlit")

# Exibir o código completo
with st.expander("Ver Código Completo"):
    st.code('''
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, explained_variance_score
import numpy as np
import tensorflow as tf

# Código adaptado aqui...
''')

# Carregar os dados
st.header("Dados e Pré-processamento")
uploaded_file = st.file_uploader("Faça upload do dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Visualização dos Dados:")
    st.write(df.head())

    # Separar features e alvo
    X = df.drop(columns=['Município', 'Média_permanência', 'Valor_médio_intern', 'Taxa_mortalidade', 'Valor_total', 'Internações'])
    y = df['Valor_médio_intern']

    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Escalonamento
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    # Construir e treinar o modelo
    input_dim = X_train_scaled.shape[1]
    output_dim = 1
    model = criar_modelo(input_dim, output_dim)

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

    st.subheader("Treinamento do Modelo")
    with st.spinner("Treinando o modelo..."):
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=1000,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stop]
        )

    # Predições e métricas
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    variance_score = explained_variance_score(y_test, y_pred)

    st.subheader("Resultados")
    st.write(f"**MAPE:** {mape*100:.3f}%")
    st.write(f"**MAE:** {mae:.3f}")
    st.write(f"**Explained Variance Score:** {variance_score:.3f}")

    # Gráficos de Perda
    st.subheader("Gráfico de Perda")
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Loss (treinamento)')
    plt.plot(history.history['val_loss'], label='Loss (validação)')
    plt.title('Perda de Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    st.pyplot(plt)
