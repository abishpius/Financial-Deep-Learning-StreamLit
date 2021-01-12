import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Simple Time Series Calculator")
st.markdown("This application is a streamlit LSTM prediction model that takes in single array time series data and returns prediction MAE on the final portion")

file_up = st.file_uploader("Upload your time series", type="CSV")
if file_up is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(file_up)
    st.write(dataframe)

series = np.asarray(dataframe.columns, dtype=float)


time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
def plot_series(time, series, format="-", start=0, end=None):
    plt.figure(figsize=(10, 6))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
    

if st.button('Plot Full Data'):
    fig = plot_series(time, series)
    st.write("Full Data")
    st.pyplot(fig)
    

if st.button("Validation Split"):
    fig2 = plot_series(time_valid, x_valid)
    st.write("Chosen Validation Split")
    st.pyplot(fig2)

if st.button("Let's Start Training!"):
    st.write("Training the LSTM model, give it a sec...")
    tf.keras.backend.clear_session()
    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                          input_shape=[None]),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(1),
      tf.keras.layers.Lambda(lambda x: x * 100.0)
    ])


    model.compile(loss="mae", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
    history = model.fit(dataset,epochs=50)
    
    st.write("Plotting the results....")
    forecast = []
    results = []
    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]
        
    f1 = plot_series(time_valid, results)
    st.pyplot(f1)
    
    
    p = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
    st.markdown("## Mean Absolute Error: %s" % (p))

    st.write("Review the Results")
    
    #mae=history.history['mae']
    loss=history.history['loss']

    epochs=range(len(loss)) 
    def plot2():
        #plt.plot(epochs, mae, 'r')
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, 'b')
        plt.title('Training Loss')
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        plt.legend(["Loss"])
        plt.grid(True)
        plt.show()
        
    q = plot2()
    st.pyplot(q)
        


