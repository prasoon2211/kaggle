import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout 
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import GridSearchCV

width = 28
height = 28
channels = 1
sample_sz = 500
DEFAULT_DROPOUT = 0.3
tune = False
train_new = True
wts_filepath = "wts-best.hdf5"

def preproc_data(X):
    X = X.astype('float32')
    X -= np.mean(X, axis=0)
    X /= (np.std(X, axis=0) + 1e-4)
    X = X.reshape((X.shape[0], height, width, channels))
    return X

def make_model(dropout=DEFAULT_DROPOUT, load=False):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(height, width, channels), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))
    
    if load:
        model.load_weights(wts_filepath)
        
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


# data
train = pd.read_csv("train.csv")
y_train = to_categorical(train.iloc[:, 0])
x_train = preproc_data(train.iloc[:, 1:].values)

test = pd.read_csv("test.csv")
x_test = preproc_data(test.iloc[:, :].values)

# tune
if tune:
    sample_idx = np.random.choice(x_test.shape[0], size=sample_sz, replace=False)
    y_sample = y_train[sample_idx, :]
    x_sample = x_train[sample_idx, :]

    param_grid = {'epochs': [10, 20, 50],
                  'batch_size': [10, 50, 100],
                  'dropout': [0.1, 0.3]}
    model = KerasClassifier(build_fn=make_model, verbose=1)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)  
    grid_result = grid.fit(x_sample, y_sample)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

# test
if train_new:
    clf = make_model(dropout=0.5)
    checkpoint = ModelCheckpoint(wts_filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='max')
    clf.fit(x_train, y_train, epochs=50, batch_size=50, validation_split=0.2, callbacks=[checkpoint])
    clf = make_model(dropout=0.1, load=True)
else:
    # load best model
    clf = make_model(dropout=0.1, load=True)

y_test = clf.predict_classes(x_test)
pred_df = pd.DataFrame()
pred_df["Label"] = y_test
pred_df["ImageId"] = list(range(1, y_test.shape[0] + 1))

pred_df.to_csv("pred.csv", index=False)
