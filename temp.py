import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import array, setdiff1d, where

raw = pd.read_excel('Indicators20190210.xlsx')
raw['year'] = raw['Date'].dt.year
raw['month'] = raw['Date'].dt.month
raw['day'] = raw['Date'].dt.day
raw['FuturesReturn(%)'] = raw['FuturesReturn(%)']/100
raw = raw.drop(columns = ['Date', 'FuturesCP', 'TSECP', 'Clear Date']).iloc[1:, :]
x, y = raw.drop(columns = ['FuturesReturn(%)']), raw['FuturesReturn(%)']

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
data = pd.concat([pd.DataFrame(y.values), pd.DataFrame(x)], axis = 1, ignore_index = True).values

def split_sequences(sequences, n_steps_in, n_steps_out):   
    X, y = [], []
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i : end_ix, 1:], sequences[end_ix : out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

n_steps_in, n_steps_out = 10, 1
x, y = split_sequences(data, n_steps_in, n_steps_out)
y = array([where(y>0.015, 2, where(y<-0.015, 1, 0))])[0, :, :]
n_output = array(y.shape[1:]).prod()
y = y.reshape((y.shape[0], n_output))
n_features = x.shape[2]
id0 = array(sum([list(range(61+100*i, 71+100*i)) for i in range(int(raw.shape[0]/100))], []))
id1 = setdiff1d(array(list(range(x.shape[0]))), id0)
train_y, test_y, train_x, test_x = y[id1], y[id0], x[id1, :, :], x[id0, :, :]
way = 0
def cnn_ts_batchnormal_way1(filter0, rate0, unit0, b_size):
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
    from keras.layers.convolutional import Conv1D, MaxPooling1D
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    model = Sequential()
    model.add(Conv1D(filters = filter0, kernel_size = 2, input_shape = (n_steps_in, n_features)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate = rate0))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(units = unit0, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))
    adam = Adam(lr = 0.001)  
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    callback = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1, mode = 'auto')
    if way == 0:
        model.fit(train_x, np_utils.to_categorical(train_y), epochs = 1000, batch_size = b_size, validation_split = 0.20, callbacks = [callback])
        loss, acc = model.evaluate(train_x, np_utils.to_categorical(train_y))
    elif way == 1:
        model.fit(train_x, np_utils.to_categorical(train_y), epochs = 1000, batch_size = b_size, callbacks = [callback])
        loss, acc = model.evaluate(train_x, np_utils.to_categorical(train_y))
    return [acc, model]

from skopt.space import Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize

space = [
        Categorical([50, 45, 35], name = 'filter0'), 
        Real(0, 0.3, name = 'rate0'),
        Categorical([64, 128, 256], name = 'unit0'),
        Categorical([32, 64, 128], name = 'b_size')
        ]

@use_named_args(space)
def objective0(**params):
    model = cnn_ts_batchnormal_way1(**params)
    return -model[0]

res0 = gp_minimize(objective0, space, n_calls = 30, acq_func = 'EI', n_points = 100000, n_jobs = -1)
obj = res0.fun
best = res0.x
param_name = ['filter0', 'rate0', 'unit0', 'b_size']
para_dict = {param_name[i]: best[i] for i in range(len(param_name))}
way = 1
final = cnn_ts_batchnormal_way1(**para_dict)
yhat = final[1].predict(test_x).argmax(axis = 1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print(confusion_matrix(test_y, yhat))
print(accuracy_score(test_y, yhat))
print(f1_score(test_y, yhat, average = 'micro'))