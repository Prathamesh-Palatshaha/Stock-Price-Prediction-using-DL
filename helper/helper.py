import numpy as np
# Dataset generation function

def createDataset(data, prediction_days):
    X_train = []
    Y_train = []
    for i in range(prediction_days, len(data)):
        X_train.append(data[i-prediction_days:i, 0])
        Y_train.append(data[i,0])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1],1))
    return np.array(X_train), np.array(Y_train)

# Models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def createLstmModel(neurons_array, inpShape, dropout):
    model = Sequential()
    #Input Layer
    model.add(LSTM(neurons_array[0], return_sequences=True, input_shape=inpShape))
    model.add(Dropout(dropout))
    #Hidden Layer 1
    model.add(LSTM(neurons_array[1], return_sequences=True))
    model.add(Dropout(dropout))
    #Hidden Layer 2
    model.add(LSTM(neurons_array[2]))
    model.add(Dropout(dropout))
    #Output Layer
    model.add(Dense(1))

    model.compile(optimizer ='adam', loss='mean_squared_error')
    # model.fit(xtrain, ytrain, epochs=30, batch_size=100)
    return model

def lstmModel(xtrain, ytrain):
    model = KerasRegressor(build_fn=createLstmModel, verbose=0)
    neurons_array = [(50, 50, 50), (64, 64, 64)]
    dropout = [0.2, 0.3]
    param_grid = dict(neurons_array=neurons_array, dropout=dropout)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(xtrain, ytrain)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result.best_estimator_
