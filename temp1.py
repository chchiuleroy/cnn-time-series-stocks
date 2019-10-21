import pandas as pd

raw = pd.read_excel('C:/Users/user/Desktop/Indicators20190210.xlsx')
raw['year'] = raw['Date'].dt.year
raw['month'] = raw['Date'].dt.month
raw['day'] = raw['Date'].dt.day
raw['FuturesReturn(%)'] = raw['FuturesReturn(%)']/100
raw = raw.drop(columns = ['Date', 'FuturesCP', 'TSECP', 'Clear Date']).iloc[1:, :]
x, y = raw.drop(columns = ['FuturesReturn(%)']), raw['FuturesReturn(%)']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
data = pd.concat([pd.DataFrame(y.values), pd.DataFrame(x)], axis = 1, ignore_index = True).values

from numpy import array, setdiff1d, where, unique
    
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
train_xn, test_xn = train_x.reshape(train_x.shape[0], n_steps_in, n_features, 1), test_x.reshape(test_x.shape[0], n_steps_in, n_features, 1)
threshold = unique(train_y, return_counts = True)[1]/train_y.shape[0]

def cnn2Dmodel(filter0, filter1, rate0, rate1, rate2, unit0):
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    model = Sequential()
    model.add(Conv2D(filters = filter0, kernel_size = (2, 2), padding = 'same', input_shape = (n_steps_in, n_features, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))    
    model.add(Dropout(rate = rate0))
    model.add(Conv2D(filters = filter1, kernel_size = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))    
    model.add(Dropout(rate = rate1))   
    model.add(Flatten())
    model.add(Dense(units = unit0, activation = 'relu'))
    model.add(Dropout(rate = rate2))
    model.add(Dense(3, activation = 'softmax'))
    return model

def model_fit(model, x_train, y_train, x_valid, y_valid, b_size):
    from keras.utils import np_utils
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    adam = Adam(lr = 0.001)  
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    callback = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1, mode = 'auto')
    model.fit(x_train, np_utils.to_categorical(y_train), epochs = 1000, batch_size = b_size, validation_data = (x_valid, np_utils.to_categorical(y_valid)), 
              callbacks = [callback])
    loss, acc = model.evaluate(x_train, np_utils.to_categorical(y_train))
    return acc

from skopt.space import Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize

space = [
        Categorical([16, 10, 8, 6], name = 'filter0'), 
        Categorical([32, 20, 16, 10], name = 'filter1'),
        Real(0, 0.5, name = 'rate0'),
        Real(0, 0.5, name = 'rate1'),
        Real(0, 0.5, name = 'rate2'),
        Categorical([64, 128, 256], name = 'unit0'),
        Categorical([32, 64, 128], name = 'b_size')
        ]

@use_named_args(space)
def objective0(filter0, filter1, rate0, rate1, rate2, unit0, b_size):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = 3)
    for train_index, valid_index in skf.split(train_xn, train_y):
        x_train, x_valid = train_xn[train_index], train_xn[valid_index]
        y_train, y_valid = train_y[train_index], train_y[valid_index]
        model = cnn2Dmodel(filter0, filter1, rate0, rate1, rate2, unit0)
        fitting = model_fit(model, x_train, y_train, x_valid, y_valid, b_size) 
    return -fitting

res0 = gp_minimize(objective0, space, n_calls = 10, acq_func = 'EI', n_points = 100000, n_jobs = -1)
obj = res0.fun
best = res0.x
model = cnn2Dmodel(*best[0:6])
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
adam = Adam(lr = 0.001)  
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
callback = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1, mode = 'auto')
model.fit(train_xn, np_utils.to_categorical(train_y), epochs = 1000, batch_size = best[-1], callbacks = [callback])
yhat = model.predict(test_xn)
y_pred = (yhat-threshold).argmax(axis = 1)
y_pred1 = yhat.argmax(axis = 1)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
print(confusion_matrix(test_y, y_pred), confusion_matrix(test_y, y_pred1))
print(accuracy_score(test_y, y_pred), accuracy_score(test_y, y_pred1))
print(f1_score(test_y, y_pred, average = 'weighted'), f1_score(test_y, y_pred1, average = 'weighted'))