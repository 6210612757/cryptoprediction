#data module
import pandas as pd
import ssl
import matplotlib.pyplot as plt
#import os

#data munipulate module
import numpy as np
from sklearn.preprocessing import MinMaxScaler #StandardScaler


#model module
import tensorflow as tf

filepath = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"

n_input = 30 #Number of data(Days) to predict the next Day
n_output = 1 #Predict 1 Day
batch_size = 32 #Less batch lower loss
epochs = 1000
fold = 4

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
    data_test = data[int(data_len*0.9):] # Use in final eval
    print("data_train :",len(data_train))
    print("data_test :",len(data_test))

    # X_train = data_train[:int(len(data_train)*0.9)]
    # Y_val = data_train[int(len(data_train)*0.9):]

    # print('X_train',len(X_train))
    # print('Y_val',len(Y_val))

    date_train = date[:int(data_len*0.9)]
    date_test = date[int(data_len*0.9):]
    # date_X = date_train[:int(len(data_train)*0.9)]
    # date_Y = date_train[int(len(data_train)*0.9):]

    scaler = MinMaxScaler()
    X_train = np.reshape(data_train,(-1,1))
    Y_val = np.reshape(data_test,(-1,1))

    #print(X_train)
    scaler.fit(X_train)
    scaled_train = scaler.transform(X_train)
    scaled_val = scaler.transform(Y_val)
    
    #Generator
    train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(scaled_train,scaled_train,length=n_input,batch_size=batch_size)
    val_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(scaled_val,scaled_val,length=n_input,batch_size=batch_size)
    #Trainning
    model = create_model()

    checkpoint_dir = "model/f%d" % fold
    checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                verbose=0, monitor='val_loss',
                save_weights_only=True,save_best_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir="./logs/f%d" % fold,
                update_freq="epoch") 

    history = model.fit(
        train_generator,validation_data = val_generator,
        epochs=epochs,
        callbacks=[cp_callback,tb_callback]
        )
        
    print(model.summary())
    #load best model to eval
    #model.load_weights(checkpoint_path)

    #save lastest model
    model.save_weights(checkpoint_dir)

    #plot_acc(history,len(history.history['loss']),fold)

    val_pred = []
    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1,n_input,n_output))
    for i in range(len(Y_val)) :
        current_pred = model.predict(current_batch)[0]
        val_pred.append(current_pred)
        current_batch = np.append(current_batch[:,n_output:,:],[[scaled_val[i]]],axis=1)
    
    real_pred = scaler.inverse_transform(val_pred)
    print(real_pred)
    plot_2data_time(Y_val,real_pred,date_test,title = 'end')
    ###################################Shutdown PC
    #os.system("shutdown /s /t 30")

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128,activation='relu',input_shape = (n_input,n_output),return_sequences=True),
        tf.keras.layers.LSTM(64,activation='relu',return_sequences=True),
        tf.keras.layers.LSTM(32,activation='relu',return_sequences=False),
        tf.keras.layers.Dense(n_output)
        ])

    model.compile(optimizer='adam',
              loss='mean_squared_error')

    return model
def plot_2data_time(real,pred,time,title = '') :
    plt.figure()
    plt.plot(time,real,color = 'blue',label = 'Real')
    plt.plot(time,pred,color = 'red',label = 'Predicted')
    plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.savefig("plot_validate_"+title+"_f"+str(fold)+".jpg")
    
def plot_acc(history,epochs,fold) :
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Accuracy : Fold '+str(fold))

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss : Fold '+str(fold))
    plt.savefig("plot_acc"+str(fold)+".jpg")
if __name__ == "__main__":
    main()