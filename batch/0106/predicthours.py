from keras.models import Model, Sequential
from keras.layers import Input, Dense, concatenate, Embedding, LSTM, Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import np_utils, plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# const---------------------------------------------------------
predict_hours = 1
input_params = ["measured", "msm", "kishocho"]
# const---------------------------------------------------------

# utils---------------------------------------------------------
def searchColumn(word, cols):
    return [col for col in cols if word in col]
# utils---------------------------------------------------------

# load data---------------------------------------------------------
data_path = "../../../data/std_data.csv"
std_data = pd.read_csv(data_path, header=0)
std_data["time"] = pd.to_datetime(std_data["time"])
# load data---------------------------------------------------------

# input/label---------------------------------------------------------
data_ = std_data[searchColumn("measured", std_data.columns)].copy()
data_["time"] = std_data["time"] - timedelta(hours=predict_hours)
data_.columns = ["label_UGRD", "label_VGRD", "time"]
data = pd.merge(std_data, data_, on="time", how="left")
data = data.fillna(0)

input_cols = []
for param in input_params:
    cols = searchColumn(param, data.columns)
    input_cols.extend(cols)

label_cols = searchColumn("label", data.columns)
# input/label---------------------------------------------------------

# model---------------------------------------------------------
def custom_model_1(inputs):
    inputs_size = inputs.shape[1]
    
    inputs = Input(shape=(inputs_size,), name="inputs")
    x = Dropout(0.2)(inputs)
    
    x = Dense(inputs_size, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    predictions = Dense(1)(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer='adam', loss='mse')
    SVG(model_to_dot(model, show_shapes = True).create(prog='dot', format='svg'))
    return model
# model---------------------------------------------------------

# learning---------------------------------------------------------
def custom_fit(model, inputs, label, validation_data):
    # callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    history = model.fit(np.array(inputs), np.array(label),
                        batch_size=4096,
                        epochs=200,
                        verbose=1,
                        validation_data=validation_data,
                        # callbacks=[callback]
                       )
    return (model, history)


def plot_history(history, file_name, min_loss):
    # 損失の履歴をプロット
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('test mse: {0}'.format(min_loss))
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig('{0}.jpg'.format(file_name))


start_2017 = datetime(2017,7,1,0,0,0)
train_data = data.query('time < \"{0}\" '.format(start_2017))
test_data = data.query('time >= \"{0}\" '.format(start_2017))

train_inputs = train_data[input_cols]
train_label = train_data[label_cols[0]]
test_inputs = test_data[input_cols]
test_label = test_data[label_cols[0]]

# 基準のmse
measured_cols = searchColumn("measured", test_data.columns)
measured_data = test_data[measured_cols[0]]
measured_loss = mean_squared_error(measured_data, test_label)

# 1
model_name = "2hour"
min_loss = 1e10
for i in range(1,2):
    model = custom_model_1(train_inputs)
    model, history = custom_fit(model, train_inputs, train_label, ((np.array(test_inputs)), (np.array(test_label))))
    loss = mean_squared_error(model.predict(np.array(test_inputs)), test_label)
    if loss < min_loss:
       min_loss = loss
       best_model = model
       best_history = history
plot_history(best_history, model_name, min_loss)
print("measured_loss:")
print(measured_loss)
# learning---------------------------------------------------------











