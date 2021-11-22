
import math
import streamlit as st
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, MaxPooling1D, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers import LSTM
from keras.models import Sequential
import seaborn as sns
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# seed = 9
tf.random.set_seed(9)

st.title("Stock Market Prediction")

st.write("""
# Explor Different Model
Which one is best?
""")

dataset = st.sidebar.file_uploader("Upload DataSet", type=["csv", "txt"])
section = st.sidebar.selectbox("Section", ("Deep Learning", "Machine Learning"))

if dataset is not None:
    df = pd.read_csv(dataset)
    st.dataframe(df)
    st.write("Dataset Shape: ", df.shape)
    data_file_name = dataset.name


    def load_data(data):
        file_name = 'data/' + data
        file = open(file_name, 'r').readlines()[1:]
        return file


    original_data = load_data(data_file_name)

    if section == "Deep Learning":
        model_name = st.sidebar.selectbox("Select Model", ("CNN", "LSTM"))
        window_size = 10

        def preProcess_Data2(data):
            raw_data = []
            for line in data:
                try:
                    close_price = float(line.split(',')[7])
                    raw_data.append(close_price)
                except:
                    continue
            sample = len(raw_data)
            time_series = np.atleast_2d(raw_data)
            time_series = time_series.T
            # time_series = np.asarray(time_series)
            X = np.array([time_series[i:i + window_size] for i in range(0, (sample - window_size))])
            X = np.atleast_3d(X)
            Y = time_series[window_size:]
            return X, Y, sample


        X, Y, sample = preProcess_Data2(original_data)

        # split train test data
        test_size = int(0.1 * sample)
        # x_train, x_test, y_train, y_test = X[:-test_size], X[-test_size:], Y[:-test_size], Y[-test_size:]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)


        def CNN_MODEL(window_size):
            model = Sequential((
                Conv1D(input_shape=(window_size, 1), kernel_size=4, activation="relu", filters=128, padding='causal'),
                MaxPooling1D(),
                # Dropout(0.2),

                Conv1D(kernel_size=4, activation="relu", filters=128, padding='causal'),
                # MaxPooling1D(2, 2),
                # Dropout(0.2),

                Flatten(),
                # Dense(128, activation='relu'),
                # Dropout(0.2),
                Dense(1, activation='linear'),
            ))
            return model


        CNN_MODEL = CNN_MODEL(window_size)


        def LSTM_MODEL():
            model = Sequential()
            model.add(LSTM(64, activation='sigmoid', return_sequences=True, input_shape=x_train.shape[1:]))
            model.add(LSTM(64, activation='relu'))
            model.add(Dense(1))

            return model


        LSTM_MODEL = LSTM_MODEL()


        def compile_model(model):
            # opt = Adam(lr=1e-3, decay=1e-3 / 200)
            model.compile(loss='mae', optimizer='adam')
            return model


        def fit_model(model_name):
            model = []
            if model_name == "CNN":
                st.write("CNN MODEL")
                model = compile_model(CNN_MODEL)
                model.fit(x_train, y_train, epochs=8, batch_size=32, validation_split=0.2)
                score = model.evaluate(x_test, y_test)
                # st.write("Mean Squared Error:", score[0])
                st.write("Mean Absolute Error:", score)

            elif model_name == "LSTM":
                st.write("LSTM MODEL")
                model = compile_model(LSTM_MODEL)
                model.fit(x_train, y_train, epochs=8, batch_size=32, validation_split=0.2)
                score = model.evaluate(x_test, y_test)
                # st.write("Mean Squared Error:", score[0])
                st.write("Mean Absolute Error:", score)

            return model


        fit_model = fit_model(model_name)

        pred = fit_model.predict(x_test)

        accuracy = r2_score(y_test, pred)
        st.write("Accuracy: ", accuracy * 100)
        col1, col2 = st.beta_columns(2)
        col1.success('Actual Value')
        col1.write(y_test)
        col2.success('Prediction Value')
        col2.write(pred)

        st.write("Here is our chart")
        fig = plt.figure(figsize=(100, 50))
        sns.set(rc={"lines.linewidth": 4})
        plt.plot(y_test[[range(0, 100)]], color='red', label='Actual Price')
        sns.set(rc={"lines.linewidth": 2})
        plt.plot(pred[[range(0, 100)]], color='green', label='predicted Price')
        # plt.title('Stock price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)
        # plt.show()

    elif section == "Machine Learning":
        model_name = st.sidebar.selectbox("Select Model", ("Decision Tree", "Random Forest"))

        # if dataset is not None:
        #     df = pd.read_csv(dataset)
        #     st.dataframe(df)
        #     st.write(df.shape)
        sample = df.shape[0]

        dummie = pd.get_dummies(df.TRADING_CODE)
        marge = pd.concat([df, dummie], axis=1)
        final_df = marge.drop(['TRADING_CODE'], axis=1)
        final_df = final_df.replace('[^\d.]', '', regex=True).astype(float)

        x = final_df.drop(['CLOSEP*'], axis=1)
        y = final_df.filter(['CLOSEP*'])
        test_size = int(0.1 * sample)
        # xtrain, xtest, ytrain, ytest = x[:-test_size], x[-test_size:], y[:-test_size], y[-test_size:]


        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=1)


        def random_forest():
            model = RandomForestRegressor(random_state=1, n_estimators=8)
            return model


        random_forest = random_forest()


        def decision_tree():
            model = DecisionTreeRegressor(random_state=1, criterion='mse')
            return model


        decision_tree = decision_tree()


        def fit_model(model_name):
            model = []
            # score = []
            if model_name == "Random Forest":
                st.write(model_name)
                model = random_forest.fit(xtrain, ytrain)
                # score = random_forest.score(xtest, ytest)
                # st.write("Accuracy:", score * 100)
            elif model_name == "Decision Tree":
                st.write(model_name)
                model = decision_tree.fit(xtrain, ytrain)
                # score = decision_tree.score(xtest, ytest)
                # st.write("Accuracy: ", score * 100)
            return model


        fit_model = fit_model(model_name)
        pred = fit_model.predict(xtest)
        # st.write('Mean Squared Error:', metrics.mean_squared_error(ytest, pred))
        st.write('Mean Absolute Error:', metrics.mean_absolute_error(ytest, pred))
        # st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, pred)))
        accuracy = r2_score(ytest, pred)
        st.write("Accuracy: ", accuracy * 100)
        col1, col2 = st.beta_columns(2)
        col1.success('Actual Value')
        ytest = np.array(ytest)
        col1.write(ytest)
        col2.success('Prediction Value')
        col2.write(pred)

        st.write("Here is our chart")
        fig = plt.figure(figsize=(100, 50))
        sns.set(rc={"lines.linewidth": 9})
        plt.plot(ytest[[range(0, 100)]], color='red', label='Actual Price')
        sns.set(rc={"lines.linewidth": 5})
        plt.plot(pred[[range(0, 100)]], color='green', label='predicted Price')
        # plt.title('Stock price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)
        # plt.show()
