# Financial-Deep-Learning-StreamLit

[Check out the APP](https://share.streamlit.io/abishpius/financial-deep-learning-streamlit/main/Time_Series_APP.py)

This web app uses an LSTM model to predict any time series model range given an input of data (in csv format) in two columns with time in the first column and values in the second column. There should be no headers in your CSV file, see submission example in the repository.
<br>
<br>
How it works: <br>
1. The application will generate a three hidden layer LSTM model (unoptimized) using the first 3/4 of the data, so depending on data size this may take a while. <br>
2. It will make predictions and output a graph plus MAE (mean absolute error) on the remaining 1/4 of the data. <br>
<br>
To learn more on deep learning with financial time series: <br>
https://colab.research.google.com/drive/15e91LVLMCNzkH3LSBrVFruQF1HxaWPFI?usp=sharing

<br>
<br>
Features to come: <br>
Ability to export the model weights.
