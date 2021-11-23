#data module
import pandas as pd
import ssl
import matplotlib.pyplot as plt

#data munipulate module
import numpy as np
from sklearn.preprocessing import MinMaxScaler #StandardScaler


#model module
import tensorflow as tf
from lstm_train import create_model
from lstm_train import plot_2data_time

filepath = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"
model_PATH = "D:\Downloads\DeepResearch\CryptoPredictionLSTM\model"
fold = 3
epoch = 459
n_input = 30 #Number of data(Days) to predict the next Day
n_output = 1 #Predict 1 Day


def main() :
    #Data getting
    ssl._create_default_https_context = ssl._create_unverified_context
    #Data timestamp is in us timezone 00.00 am = 12.00 pm
    df = pd.read_csv(filepath, skiprows=1)  # we use skiprows parameter because first row contains our web address
    #print(df.iloc[1459]['close'])  # approx. btc 1460 data start from 2017-08-17 till now

    #data munipulate
    data = df.close #Use close as feature
    date = df.date
    data = np.flip(data.to_numpy())
    date = np.flip(date.to_numpy())
    data_len = len(data)

    data_train = data[:int(data_len*0.9)]
    date_train = date[:int(data_len*0.9)]

    data_test = data[int(data_len*0.9):]
    date_test = date[int(data_len*0.9):]

    scaler = MinMaxScaler()
    X_train = np.reshape(data_train,(-1,1))
    Y_val = np.reshape(data_test,(-1,1))

    scaler.fit(X_train)
    scaled_train = scaler.transform(X_train)
    scaled_val = scaler.transform(Y_val)

    model = create_model()
    model.load_weights(model_PATH+f"/f{fold}"+f"/cp-{epoch:04d}.ckpt")

    val_pred = []
    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1,n_input,n_output))
    for i in range(len(Y_val)) :
        current_pred = model.predict(current_batch)[0]
        val_pred.append(current_pred)
        current_batch = np.append(current_batch[:,n_output:,:],[[scaled_val[i]]],axis=1)
    
    real_pred = scaler.inverse_transform(val_pred)
    print(real_pred)
    plot_2data_time(Y_val,real_pred,date_test,title='Test'+str(epoch))

if __name__ == "__main__":
    main()