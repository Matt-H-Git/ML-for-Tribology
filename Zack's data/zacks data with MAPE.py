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
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, ConvLSTM1D, Conv2D, Conv2DTranspose, Dropout, Input, GroupNormalization, Dense, TimeDistributed, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ============= Utils ============= #


def get_dataset(file_path, num_batches, regex_list, data_type='train'): #For multiple batches of files - For data_type='train', num_batches is in the range [1,4]
    if num_batches == 0:
        raise ValueError('get_dataset:: Must have at least one batch in the dataset') 
    if (data_type == 'test') and (num_batches > 2):
        raise ValueError('get_dataset:: Only 2 batches of test data available right now')
    if (data_type == 'train') and (num_batches > 4):
        raise ValueError('get_dataset:: Only 4 batches of training data available right now')
    #if len(regex_list) != num_batches:
    #    raise ValueError('get_dataset:: length of regex list not equal to number of batches - either provide different number of regex endings or change num_batches')
    #data_type can be 'train' or 'test'. This effects the regex that is searched for, and will need to be manually checked if more test folder are provided
    #Regex list provides a list of file endings to search for to find each batch of data. Its length should be equal to num_batches
    #For data_type == 'test' this still applies - but only the (num_batches-1) index of regex list will be used
    file_names = glob.glob(file_path)
    data = []
    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    global regex_matches
    global num_features
    regex_matches = 0
    num_features = (0, 4) #Used to vary the number of features selected in the data
    def regex(to_match, file, data):
        global regex_matches
        global num_features
        regex_result = len(re.findall(fr'\\*_[0-9]{{3}}({to_match})', file))
        if regex_result == 1:
            regex_matches += 1
            df = pd.read_csv(file, header=None)
            data = np.append(data, df.iloc[:, num_features[0]:num_features[1]].to_numpy())
        elif regex_result > 1:
            print(file)
            raise ValueError('More than one regex match in the above string was found')
        return data
    
    def re_shape(data, batches):
        global regex_matches
        global num_features
        seq_len = int(len(data)/(regex_matches*num_features)) #number of samples in one excel file (per feature)
        return np.reshape(data, (regex_matches, seq_len, num_features))
        
    for file in file_names:
        #Use regex to filter repeated files (just use the first sample of each time point for now)
        if data_type == 'train':
            match num_batches:
                case 1:
                    data_1 = regex(regex_list[0], file, data_1)
                case 2:
                    data_1 = regex(regex_list[0], file, data_1)
                    data_2 = regex(regex_list[1], file, data_2)
                case 3:
                    data_1 = regex(regex_list[0], file, data_1)
                    data_2 = regex(regex_list[1], file, data_2)
                    data_3 = regex(regex_list[2], file, data_3)
                case 4:
                    data_1 = regex(regex_list[0], file, data_1)
                    data_2 = regex(regex_list[1], file, data_2)
                    data_3 = regex(regex_list[2], file, data_3)
                    data_4 = regex(regex_list[3], file, data_4)
        elif data_type == 'test':
            match num_batches:
                case 1:
                    data = regex(regex_list[num_batches-1], file, data)
                case 2:
                    data = regex(regex_list[num_batches-1], file, data)
    num_features = num_features[1]-num_features[0]
    if data_type == 'train':
        regex_matches = int(regex_matches / num_batches)
        match num_batches:
            case 1:
                data_1 = np.expand_dims(re_shape(data_1, num_batches), 0)
                data = data_1
            case 2:
                data_1 = np.expand_dims(re_shape(data_1, num_batches), 0)
                data_2 = np.expand_dims(re_shape(data_2, num_batches), 0)
                data = np.concatenate((data_1, data_2), axis=0)
            case 3:
                data_1 = np.expand_dims(re_shape(data_1, num_batches), 0)
                data_2 = np.expand_dims(re_shape(data_2, num_batches), 0)
                data_3 = np.expand_dims(re_shape(data_3, num_batches), 0)
                data = np.concatenate((data_1, data_2, data_3), axis=0)
            case 4:
                data_1 = np.expand_dims(re_shape(data_1, num_batches), 0)
                data_2 = np.expand_dims(re_shape(data_2, num_batches), 0)
                data_3 = np.expand_dims(re_shape(data_3, num_batches), 0)
                data_4 = np.expand_dims(re_shape(data_4, num_batches), 0)
                data = np.concatenate((data_1, data_2, data_3, data_4), axis=0)
    elif data_type == 'test':
        # print(regex_matches) Use to check no. timesteps in the selected test data
        data = np.expand_dims(re_shape(data, num_batches), 0)
        
    if data_type == 'test':
        return data[:, 30:60, :30000, 1:] # Change the second dimension here to choose which part of the timestep you want to test and view
    else:
        return data[:, :30, :30000, 1:] #Remove the first feature from last axis of data, too different to the other features

def preprocess_old(data):
    #data must be in shape NxSxF for N number of excel files, S samples (rows) in each file and f features (columns) in each file
    scaler = StandardScaler()
    #Normalise the data - need to do this feature by feature
    scaled = []
    for i in range(data.shape[2]):
        scaled.append(scaler.fit_transform(data[:, :, i]))
    return np.reshape(scaled, data.shape)

def preprocess(train_data, test_data):
    print('Preprocessing data')
    #train data must be in shape BxTxSxF for B batches, T timesteps (excel files), S samples (rows) in each file and f features (columns) in each file
    #Test data must be in shape TxSxF - will be output as 1xTxSxF
    if len(train_data.shape) != 4:
        raise ValueError(f'preprocess::train_data has {len(train_data.shape)} dimensions - must have 4 dimensions')
    if len(test_data.shape) != 3:
        raise ValueError(f'preprocess::test_data has {len(test_data.shape)} dimensions - must have 3 dimensions')
    #Preprocessing algorithm
    #Take average of training batches
    #Create a different scaler for each feature (this scaler takes maximum two dimensions)
    #fit on the training average, then transform every feature of training and test data on the respective scaler

    #Average
    averaged = np.mean(train_data, axis=0)
    """
    for i in range(train_data.shape[3]): #Visualise average compared to original data
        print(f'Viewing feature {i}')
        view_results(train_data[3, :, :, i], averaged[:, :, i])
    """
    #Create scalers and fit on training avg
    train_sc = []
    test_sc = []
    for i in range(averaged.shape[2]):
        scaler = StandardScaler().fit(averaged[:, :, i])
        #scaler = StandardScaler().fit(train_data[0, :, :, i])
        #Scale the remaining data
        for j in range(train_data.shape[0]):
            train_sc = np.append(train_sc, scaler.transform(train_data[j, :, :, i]))
        test_sc = np.append(test_sc, scaler.transform(test_data[:, :, i]))
    train_sc = np.reshape(train_sc, train_data.shape)
    test_sc = np.reshape(test_sc, test_data.shape)
    test_sc = np.expand_dims(test_sc, axis=0)
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
        ConvLSTM1D(filters=64, kernel_size=7, padding='same', strides=2, return_sequences=True, activation="tanh", recurrent_dropout=0.2),
        #Conv2D(filters=64, kernel_size=5, padding='same', strides=1, activation='relu', input_shape=(num_files, num_samples, num_features)),
        ConvLSTM1D(filters=32, kernel_size=3, padding='same', strides=1, return_sequences=True, activation="tanh", recurrent_dropout=0.2),
        #Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=(num_files, num_samples, num_features)),

        Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=1, activation='tanh', input_shape=(num_files, num_samples, num_features)),
        Dropout(rate=0.2),
        Conv2DTranspose(filters=64, kernel_size=5, padding='same', strides=1, activation='tanh'),
        Conv2DTranspose(filters=num_features, kernel_size=1, strides=2, padding='same'),
        AveragePooling2D(pool_size=(1,1), strides=(2, 1), padding='valid')
        ])

    

    model_rnn = tf.keras.Sequential([ #Dropout provides no increase in learning here
        Input(shape=data.shape),
        ConvLSTM1D(filters=64, kernel_size=5, padding='same', return_sequences=True, activation="relu",),
        Dense(64, activation='relu'),
        Dropout(rate=0.2),
        ConvLSTM1D(filters=64, kernel_size=3, padding='same', return_sequences=True, activation="relu",),
        Dense(32, activation='relu'),
        Dropout(rate=0.2),
        ConvLSTM1D(filters=64, kernel_size=1, padding='same', return_sequences=True, activation="relu",),
        Dense(16, activation='relu'),
        Dropout(rate=0.2),
        Conv2D(filters=num_features, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=(num_files, num_samples, num_features)),
        ])
    model_3 = tf.keras.Sequential([
        Input(shape=np.expand_dims(data, axis=0).shape),
        TimeDistributed(Conv2D(filters=64, kernel_size=5, padding='same', strides=2, activation='relu', input_shape=(num_files, num_samples, num_features))),
        Dropout(rate=0.2),
        TimeDistributed(Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=(num_files, num_samples, num_features))),

        TimeDistributed(Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=(num_files, num_samples, num_features))),
        Dropout(rate=0.2),
        TimeDistributed(Conv2DTranspose(filters=64, kernel_size=5, padding='same', strides=2, activation='relu')),
        TimeDistributed(Conv2DTranspose(filters=num_features, kernel_size=1, padding='same')),
    ])

    model_2 = tf.keras.Sequential([
        Input(shape=data.shape),
        Conv2D(filters=128, kernel_size=5, padding='same', strides=2, activation='elu', input_shape=(num_files, num_samples, num_features)),
        Dropout(rate=0.2),
        Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation=LeakyReLU(), input_shape=(num_files, num_samples, num_features)),

        Conv2DTranspose(filters=64, kernel_size=3, padding='same', strides=1, activation=LeakyReLU(), input_shape=(num_files, num_samples, num_features)),
        Dropout(rate=0.2),
        Conv2DTranspose(filters=128, kernel_size=5, padding='same', strides=2, activation='selu'),
        Conv2DTranspose(filters=num_features, kernel_size=1, padding='same'),
        ])
    #model_2 is used to produce the results in the report
    model = model_rnn
       
    model.compile(optimizer=Adam(learning_rate=0.0015), loss='mse')
    model.summary()
    best_model.compile(optimizer=Adam(learning_rate=0.0015), loss='mse')
    #best_model.summary()
    return model

# =============================================== #

# ============= Model Training / Testing ============= #

def train(model, data, weights_path, num_epochs, visualise=0):
    batch_size=1
    history = model.fit(data, data, epochs=num_epochs, batch_size=batch_size, verbose=2, shuffle=False)
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
    data_pred = model.predict(data)
    MSE, MAPE = show_metrics(data, data_pred)
    #return 0
    feature_dict = {0:'S1', 1:'S2', 2:'vib'}
    for i in range(data.shape[3]): #Visualise results for each feature
        print(f'Viewing feature {i}')
        title_str = f'Feature {feature_dict[i]} with MSE {MSE:.2f} and MAPE {MAPE:.2f}'
        view_results(data[0, :, :, i], data_pred[0, :, :, i], title=title_str)
        #view_results(data[0, 0, :, :, i], data_pred[0, 0, :, :, i]) for timedistributed model - do not think the view_results will work with this anymore

def show_metrics(y, y_pred):
    mses = []
    mapes = []
    for i in range(y.shape[1]): # Metrics are taken as an average of the errors for each feature
        #mses.append(mean_squared_error(y[0, :, :, i], y_pred[0, :, :, i]))
        #mapes.append(mean_absolute_percentage_error(y[0, :, :, i], y_pred[0, :, :, i]))
        mses.append(mean_squared_error(y[0, i, :, :], y_pred[0, i, :, :]))
        mapes.append(mean_absolute_percentage_error(y[0, i, :, :], y_pred[0, i, :, :]))
        """
        print(mean_absolute_percentage_error(y[0, i, :, :], y_pred[0, i, :, :], multioutput='raw_values'))
        mapes = np.append(mapes, mean_absolute_percentage_error(y[0, i, :, :], y_pred[0, i, :, :]))
        print(mapes[i])
        #if mapes[i] > 100:
        #if (i == 0) or (i == 7):
        if (i == 7):
            print(f'index: {i}')
            for j in range(y.shape[3]):
                print(f'Viewing feature{j}')
                view_single(y[0], y_pred[0], index=i, feature=j)
    """
    MSE = np.average(mses)
    MAPE = np.average(mapes)
    print(f'MSE: {MSE}')
    print(f'MAPE: {MAPE}')
    return MSE, MAPE

def view_single_thing(y, index, feature):
    #Feature is input as an int from [0-3] or [0-2] depending on if encoder is included or not
    #Shape TxSxF for T timesteps, S samples and F features
    plt.plot(y[index, :, feature], 'b-', label='y')
    plt.legend()
    plt.title(f'Feature {feature}, timestep {index}')
    plt.show()
    plt.close()

def view_single(y, y_pred, index, feature):
    #Feature is input as an int from [0-3] or [0-2] depending on if encoder is included or not
    plt.plot(y[index, :, feature], 'b-', label='y')
    plt.plot(y_pred[index, :, feature], 'r-', label='y_pred')
    
    plt.legend()
    plt.title(f'Feature {feature}, timestep {index}')
    plt.show()
    plt.close()

def view_results(y, y_pred, title=None): 
    #Results are shown for a single feature at a time
    #Input shape for y and y_pred is NxS for N number of excel files and S number of samples in each file
    #Show results as 5x6 subplot
    fig, ax = plt.subplots(5, 6)
    i = 0
    y_lim = (np.minimum(y_pred.min(), y.min()), np.maximum(y_pred.max(), y.max()))
    y_lim = (round(y_lim[0], 1), round(y_lim[1], 1))
    for row in ax:
        for col in row:
            col.plot(y[i, :], 'b-')
            col.plot(y_pred[i, :], 'r-', label=(i+30))
            col.set_ylim(y_lim)
            col.legend()
            i += 1
    plt.suptitle(title)
    plt.show()
    plt.close()

# ==================================================== #



def main():
    # ===== Options ===== #
    train_bool = 0
    evaluate_bool = 1

    scale = True
    # =================== #
    #Training data and model
    file_path = "./Filtered data/TE74-base oil-test1/ES/*" #Just doing one test for now
    train_regex = ['001.csv', '011.csv', '021.csv', '031.csv']
    data = get_dataset(file_path, len(train_regex), train_regex)

    #data = np.reshape(data, (1, 1, num_files, num_samples, num_features)) #For model
    #data = np.expand_dims(data, axis=0) #Try for timedistributed model instead of above line
    
    #Test data
    file_path_test = "./Filtered data/TE74-base oil-test4/ES/*" #Just doing one test for now
    test_regex = ['00[1,2].csv', '50[1,2].csv'] #96 timesteps, 64 timesteps
    #file_path_test = file_path #For using training data as test data
    #test_regex = train_regex[3]
    test_data = get_dataset(file_path_test, 2, test_regex, data_type='test')

    if scale:
        data, test_data = preprocess(data, test_data[0])
    
    model = instantiate(data[0]) #data required for shape only, values aren't used
    #Train and test
    #weights_path = "cp.ckpt" # For normal CNN model
    weights_path = "cp_rnn.ckpt" # For RNN
    if train_bool:
        train(model, data, weights_path, num_epochs=10)
    if evaluate_bool:
        evaluate(model, test_data, weights_path)

if __name__ == '__main__':
    main()


# Backup code

# Old data retrieval code
"""
def get_dataset_old(file_path): #For one batch of files
    file_names = glob.glob(file_path)
    data = []
    regex_matches = 0
    num_features = (0, 4)
    for file in file_names:
        #Use regex to filter repeated files (jsut use the first sample of each time point for now)
        regex_result = len(re.findall(r"\\*_[0-9]{3}(001.csv)", file))
        if regex_result == 1:
            regex_matches += 1
            df = pd.read_csv(file, header=None)
            #data = np.append(data, df)
            data = np.append(data, df.iloc[:, num_features[0]:num_features[1]].to_numpy())
        elif regex_result > 1:
            print(file)
            raise ValueError('More than one regex match in the above string was found')
    num_features = num_features[1]-num_features[0]
    seq_len = int(len(data)/(regex_matches*num_features)) #number of samples in one excel file (per feature)
    data = np.reshape(data, (regex_matches, seq_len, num_features))
    #Cut off the last 3 numbers - won't make a difference and only make convolutions more difficult later on
    data = np.expand_dims(data, axis=0)
    return data[:, :30, :30000, 1:]
"""

# Old preprocess code in main()
"""
    reshaped_data = []
    for i in range(data.shape[0]):
        reshaped_data.append(preprocess(data[i]))
    data = np.reshape(reshaped_data, data.shape)
    model = instantiate(data[0]) #data required for shape only, values aren't used
"""

"""
    reshaped_data = []
    for i in range(test_data.shape[0]):
        reshaped_data.append(preprocess(test_data[i]))
    test = np.reshape(reshaped_data, test_data.shape)
"""
#Before implementing batches
"""
# ============= Model Training / Testing ============= #

"""
