import sys, re
import numpy as np , pandas as pd
import time
import glob
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, ConvLSTM1D, Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose, Dropout, Input, GroupNormalization, Dense, TimeDistributed, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ============= Utils ============= #
def get_dataset(file_path, to_match):
    file_names = glob.glob(file_path)
    data = []
    global num_features
    global regex_matches
    regex_matches = 0 
    num_features = (0, 4) #Used to vary the number of features (columns) selected in the data
    def regex(to_match, file, data):
        #to_match is the csv file ending (including the .csv)
        #file is the file name
        #data is an array, either empty or contains previously collected data for the current batch
        #regex_match is an integer for the current batch of data counting how many files have been read
        global num_features
        global regex_matches
        regex_result = len(re.findall(fr'\\*_[0-9]{{3}}({to_match})', file))
        if regex_result == 1:
            regex_matches += 1
            df = pd.read_csv(file, header=None)
            #Filtered data
            data = np.append(data, df.iloc[:, num_features[0]:num_features[1]].to_numpy())
            #Unfiltered data
            #data = np.append(data, df.iloc[4:, [1, 3, 5, 7]].to_numpy())
        elif regex_result > 1:
            print(file)
            raise ValueError('More than one regex match in the above string was found')
        return data
    
    def re_shape(data):
        global regex_matches
        global num_features
        seq_len = int(len(data)/(regex_matches*num_features)) #number of samples in one excel file (per feature)
        return np.reshape(data, (regex_matches, seq_len, num_features))

    for file in file_names:
        data = regex(to_match, file, data)
    num_features = num_features[1]-num_features[0]
    return re_shape(data)
    
def preprocess(data):
    #data must be in shape NxSxF for N number of excel files, S samples (rows) in each file and f features (columns) in each file
    scaler = StandardScaler()
    #Normalise the data - need to do this feature by feature
    scaled = []
    for i in range(data.shape[2]):
        scaled.append(scaler.fit_transform(data[:, :, i]))
    return np.reshape(scaled, data.shape)

def preprocess_new(train_data, test_data):
    print('Preprocessing data')
    #train data must be in shape TxSxF for T timesteps (excel files, also used as batches here), S samples (rows) in each file and f features (columns) in each file
    #Test data must be in shape TxSxF
    #Data is output in same shape as input
    if len(train_data.shape) != 3:
        raise ValueError(f'preprocess::train_data has {len(train_data.shape)} dimensions - must have 3 dimensions')
    if len(test_data.shape) != 3:
        raise ValueError(f'preprocess::test_data has {len(test_data.shape)} dimensions - must have 3 dimensions')
    #Preprocessing algorithm
    #Take average of training batches
    #Create a different scaler for each feature (this scaler takes maximum two dimensions)
    #fit on the training average, then transform every feature of training and test data on the respective scaler

    #Average
    averaged = np.mean(train_data, axis=0) # Produces an average timstep for each feature (shape SxF)
    
    #Create scalers and fit on training avg
    train_sc = []
    test_sc = []
    scaler = StandardScaler().fit(averaged)
    for i in range(train_data.shape[0]):
        train_sc = np.append(train_sc, scaler.transform(train_data[i, :, :]))
    for i in range(test_data.shape[0]):
        test_sc = np.append(test_sc, scaler.transform(test_data[i, :, :]))
    train_sc = np.reshape(train_sc, train_data.shape)
    test_sc = np.reshape(test_sc, test_data.shape)
    return train_sc, test_sc
# ================================= #

# ============= Model Instantiation ============= #

def instantiate(data):
    #For Thursday:
    #Change to conv2dlstm layers and see if there is improvement
    # Instantiate the ConvLSTM model, optimiser and loss functions
    num_files = data.shape[0]
    num_samples = data.shape[1]
    num_features = data.shape[2]
    
    best_model = tf.keras.Sequential([
        Input(shape=data.shape),
        Conv2D(filters=64, kernel_size=5, padding='same', strides=2, activation='selu', input_shape=(num_files, num_samples, num_features)),
        Dropout(rate=0.2),
        Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='selu', input_shape=(num_files, num_samples, num_features)),

        Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=1, activation='selu', input_shape=(num_files, num_samples, num_features)),
        Dropout(rate=0.2),
        Conv2DTranspose(filters=64, kernel_size=5, padding='same', strides=2, activation='selu'),
        Conv2DTranspose(filters=num_features, kernel_size=1, padding='same'),
        ])

    model = tf.keras.Sequential([
        Input(shape=data.shape),
        Conv2D(filters=128, kernel_size=5, padding='same', strides=2, activation='elu', input_shape=(num_files, num_samples, num_features)),
        Dropout(rate=0.2),
        Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation=LeakyReLU(), input_shape=(num_files, num_samples, num_features)),

        Conv2DTranspose(filters=64, kernel_size=3, padding='same', strides=1, activation=LeakyReLU(), input_shape=(num_files, num_samples, num_features)),
        Dropout(rate=0.2),
        Conv2DTranspose(filters=128, kernel_size=5, padding='same', strides=2, activation='selu'),
        Conv2DTranspose(filters=num_features, kernel_size=1, padding='same'),
        ])
        
    model = tf.keras.Sequential([
        Input(shape=data[0].shape),
        Conv1D(filters=128, kernel_size=5, padding='same', strides=2, activation='elu', input_shape=(num_samples, num_features)),
        Dropout(rate=0.2),
        Conv1D(filters=64, kernel_size=3, padding='same', strides=1, activation=LeakyReLU(), input_shape=(num_samples, num_features)),

        Conv1DTranspose(filters=64, kernel_size=3, padding='same', strides=1, activation=LeakyReLU(), input_shape=(num_samples, num_features)),
        Dropout(rate=0.2),
        Conv1DTranspose(filters=128, kernel_size=5, padding='same', strides=2, activation='selu'),
        Conv1DTranspose(filters=num_features, kernel_size=1, padding='same'),
        ])
        
    model.compile(optimizer=Adam(learning_rate=0.0015), loss='mse')
    model.summary()
    
    best_model.compile(optimizer=Adam(learning_rate=0.0015), loss='mse')
    best_model.summary()
    
    return model

# =============================================== #

# ============= Model Training / Testing ============= #

def train(model, data, weights_path, num_epochs, visualise=0):
    batch_size=1
    history = model.fit(data, data, epochs=num_epochs, batch_size=1, verbose=2, shuffle=False) #Maybe batch_size=None
    model.save_weights(weights_path)
    print(f'Weights saved to {weights_path}')
    if visualise:
        plt.plot(history.history["loss"], label="Training Loss")
        plt.legend()
        plt.show()

def evaluate(model, data, weights_path):
    #Load weights
    model.load_weights(weights_path)
    #Make predictions
    data_pred = []
    for i in range(data.shape[0]):
        data_pred = np.append(data_pred, model.predict(np.expand_dims(data[i, :, :], axis=0)))
    data_pred = np.reshape(data_pred, data.shape)
    MSE, MAPE = show_metrics(data, data_pred)
    feature_dict = {0:'S1', 1:'S2', 2:'vib'}
    for i in range(data.shape[2]): #Visualise results for each feature
        print(f'Viewing feature {i}')
        title_str = f'Feature {feature_dict[i]} with MSE {MSE:.2f} and MAPE {MAPE:.2f}'
        view_results(data[:, :, i], data_pred[:, :, i], title=title_str)

def show_metrics(y, y_pred):
    mses = []
    mapes = []
    for i in range(y.shape[2]): # Metrics are taken as an average of the errors for each feature
        mses.append(mean_squared_error(y[:, :, i], y_pred[:, :, i]))
        mapes.append(mean_absolute_percentage_error(y[:, :, i], y_pred[:, :, i]))
    MSE = np.average(mses)
    MAPE = np.average(mapes)
    print(f'MSE: {MSE}')
    print(f'MAPE: {MAPE}')
    return MSE, MAPE

def view_results(y, y_pred, title=None):
    #Results are shown for a single feature at a time
    #Input shape for y and y_pred is NxS for N number of excel files and S number of samples in each file
    #Show results as 5x6 subplot
    max_iter = 0
    dim_1 = 5
    dim_2 = 6

    y_lim = (np.minimum(y_pred.min(), y.min()), np.maximum(y_pred.max(), y.max()))
    y_lim = (round(y_lim[0], 1), round(y_lim[1], 1))
    
    i = 0
    while max_iter < y.shape[0]:
        fig, ax = plt.subplots(dim_1, dim_2, layout="constrained")
        #fig, ax = plt.subplots(dim_1, dim_2, layout='tight')
        #fig, ax = plt.subplots(dim_1, dim_2)
        #fig.tight_layout()
        for row in ax:
            for col in row:
                if i < y.shape[0]:
                    col.plot(y[i, :], 'b-')
                    col.plot(y_pred[i, :], 'r-', label=i)
                    col.set_ylim(y_lim)
                    col.set_title(f'Timestep {i}')
                i += 1
        max_iter += (dim_1*dim_2)
        if title != None:
            plt.suptitle(title)
        plt.show()
        plt.close()

def view_single_batch(y, feature_int):
    #Results are shown for a single feature at a time - choose which one via featuer_int in range [0,3]
    #Input shape for y and y_pred is NxS for N number of excel files and S number of samples in each file
    #Show results as 5x6 subplot
    max_iter = 0
    dim_1 = 3
    dim_2 = 6
    print(f'Viewing feature {feature_int}')
    i = 0
    print(y[0, 0, feature_int])
    while max_iter < y.shape[0]:
        fig, ax = plt.subplots(dim_1, dim_2)
        
        for row in ax:
            for col in row:
                if i < y.shape[0]:
                    col.plot(y[i, :, feature_int], 'b-')
                    col.set_title(f'Timestep {i}')
                    #col.set_ylim((-8, 8))
                    #col.axis('off')
                i += 1
                
        max_iter += (dim_1*dim_2)
        plt.show()
        plt.close()

# ==================================================== #

def main():
    
    # ===== Options ===== #
    train_bool = 0
    evaluate_bool = 1
    # =================== #
    #Training data and model
    
    file_path_train = "./Filtered data/TE74-base oil-test1/ES/*" 
    regex_test1 = ['001.csv', '011.csv', '021.csv', '031.csv']
    data = get_dataset(file_path_train, regex_test1[0])
    
    regex_test4 = ['00[1,2].csv', '50[1,2].csv']
    file_path_test = "./Filtered data/TE74-base oil-test4/ES/*"
    #file_path_test = "./Unfiltered data/TE74-base oil-test4/ES/*"
    test_data = get_dataset(file_path_test, regex_test4[0])

    data = data[:, :30000, 1:] 
    test_data = test_data[:, :30000, 1:] 

    data, test_data = preprocess_new(data, test_data)
    model = instantiate(data) #data required for shape only, values aren't used
    #Train and test
    weights_path = "cp_lower_dim.ckpt"
    
    if train_bool:
        train(model, data, weights_path, num_epochs=10)
    if evaluate_bool:
        evaluate(model, test_data, weights_path)
    

if __name__ == '__main__':
    main()


# Backup code

#Old main

"""file_path_train = "./Filtered data/TE74-base oil-test1/ES/*" 
    regex_test1 = ['001.csv', '011.csv', '021.csv', '031.csv']
    data = get_dataset(file_path_train, regex_test1[0])
    data = data[:, :30000, 1:] 
    data = preprocess(data)
    model = instantiate(data) #data required for shape only, values aren't used

    regex_test4 = ['00[1,2].csv', '50[1,2].csv']
    file_path_test = "./Filtered data/TE74-base oil-test4/ES/*"
    #file_path_test = "./Unfiltered data/TE74-base oil-test4/ES/*"
    test_data = get_dataset(file_path_test, regex_test4[0])
    test_data = test_data[:, :30000, 1:] 
    test_data = preprocess(test_data)
    #view_single_batch(test_data, feature_int=1)
"""
