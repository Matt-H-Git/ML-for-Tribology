import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler

# ============ Utils ============ #

def import_data(csv_file): # Reads in data from the csv file given
    return pd.read_csv(csv_file)
 
def preprocess(data, scaling=True, remove_std_features=False): # Removes unwanted columns
    #data = data.drop(['CyclesToFF', 'Friction Coefficient'], axis=1) 
    if remove_std_features:
        #Remove features to do with standard deviation
        headers = data.columns
        Std_indexes = []
        for i in range(len(headers)):
            if 'Std' in headers[i]:
                Std_indexes.append(i)
        data = data.drop(data.columns[Std_indexes], axis=1)
    headers = data.columns # Needed to return later - keep after the if remove_std_inp statement
    print(headers)
    end_inp = int(np.where(data.columns == 'CompRqPost')[0]) #Find the index in data headers where input ends and output begins # need to have this line after any input features are dropped
    if scaling:
        data = StandardScaler().fit_transform(data)
        input_data = data[:, 0:end_inp]
        output_data = data[:, end_inp:]
    else:
        input_data = data.iloc[:, 0:end_inp] # X
        output_data = data.iloc[:, end_inp:] # y
    return input_data, output_data, headers


def get_method(method): # Retrieve the wanted method
    if method[0]: # Random Forest
        clf = RandomForestRegressor(random_state=0)
        #{'criterion': 'absolute_error', 'max_depth': 6, 'n_estimators': 370}
    elif method[1]: # Gradient Boost
        clf= GradientBoostingRegressor(random_state=0)
    elif method[2]: # Linear Regression
        clf= LinearRegression()
    elif method[3]: # SVR
        clf= SVR()
    elif method[4]: # KNeighbours
        clf= KNeighborsRegressor()
    elif method[5]: # MLP
        clf= MLPRegressor(random_state=0)
    elif method[6]: # Gaussian Process
        clf= GaussianProcessRegressor(random_state=0)
    elif method[7]: # Decision Tree
        clf= DecisionTreeRegressor(random_state=0)
    else:
        raise ValueError("No ML method selected")
    print(clf)
    return MultiOutputRegressor(clf)

def clear_method(method): #Ensure every element in method is set to zero
    for i in range(len(method)):
        method[i] = 0
    return method

def write_back_results(result, cross_val=False): # Write back 'result' to 'dest_file'.
    # 'result' should be a Nx2 matrix for N number of models.
    dest_file = 'Multioutput Regression Results.csv'
    model_names = ["Random Forest", "Gradient Boost", "Linear Regression", "SVR", "KNeighbours", "MLP", "Gaussian Process", "Decision Tree"]
    result_dict = dict(zip(model_names, result))
    # Convert numpy array to df, attach headers then write to csv
    if cross_val:
        df = pd.DataFrame.from_dict(result_dict, orient='index', columns=['MSE Avg', 'MSE Std', 'MAPE Avg', 'MAPE Std'])
    else:
        df = pd.DataFrame.from_dict(result_dict, orient='index', columns=['MSE', 'MAPE']) 
    df.to_csv(dest_file) # Write to the csv file at location 'dest_file'
    return None

def generate_test_data(file, var_name, data_range):
    data = import_data(file)
    options = ['Pressure', 'Speed', 'SRR']
    if var_name not in options:
        raise ValueError("Incorrect input for var_name, must be 'Pressure', 'Speed' or 'SRR'.")
    #Use the input data table, take averages for all features except those in list options, for features in options keep them constant apart from the one being varied
    #Data range is the list of values the input var_name will be predicted for

    #First take averages of all the features in data
    avg = data.mean()
    #names = data.columns
    avg = avg.to_numpy()
    avg = [avg for _ in range(len(data_range))]
    avg = np.stack(avg, axis=0)
    index = int(np.where(data.columns == var_name)[0]) #Find the index in data headers which equates to var_name
    avg[:, index] = data_range#Replace the column at that index with data_range
    avg = pd.DataFrame(avg, columns=data.columns)#Make a dataframe from avg
    end_inp = int(np.where(data.columns == 'CompRqPost')[0]) #Find the index in data headers where input ends and output begins
    return avg.iloc[:, 0:end_inp]

def percentage_change(y): #Use y to create an array that shows percentage change of value compared to initial value
    #Performed column-wise, returns same shape as input
    def percent_diff(a, b): #Find the percentage difference between a and b
        return (a-b)/a
    converted = []
    initial = y[0, :]
    for i in range(y.shape[0]):
        converted = np.append(converted, percent_diff(initial, y[i, :]))
    converted = np.reshape(converted, y.shape)
    return converted

def remove_std_output_features(y, features):
    #Shape of y should be input as SxF for S samples and F features
    end_inp = int(np.where(features == 'CompRqPost')[0]) 
    features = features[end_inp:]
    indexes = []
    for i in range(len(features)):
        if 'Std' in features[i]:
            indexes.append(i)
    y = np.delete(y, indexes, axis=1)
    features = np.delete(features, indexes)
    return y, features
# ============ Graph methods ============ #

def print_results(result): # shows results as a box and whisker plot (for all clfs)
    # Show results as a box and whisker plot
    model_names = ["Random Forest", "Gradient Boost", "Linear Regression", "SVR", "KNeighbours", "MLP", "Gaussian Process", "Decision Tree"]
    result = np.reshape(result[:, 0], (1, len(result[:, 0])))
    
    plt.boxplot(result, labels=model_names)
    plt.title('Multioutput Regression Results')
    plt.show()
    plt.close()
    return None

def view_roughness(file, result, cross_val, y, features, view_metric='MSE'):
    #result can be input straight from result.append(fit_pred())
    if (view_metric != 'MAPE') and (view_metric != 'MSE'):
        raise ValueError(f'Incorrect metric selected for view_roughness ({view_metric}) - options are "MAPE" or "MSE"')
    if cross_val != False:
        raise ValueError(f'view_roughness will not work with results from cross_val = {cross_val}, must be True instead')
    
    #Get names of models and roughness measurements
    model_names = ["Random Forest", "Gradient Boost", "Linear Regression", "SVR", "KNeighbours", "MLP", "Gaussian Process", "Decision Tree"]
    
    #Reshape the data
    result = np.reshape(result, (len(result)*2, y.shape[1]))
    result = np.transpose(result)
    if len(features) != result.shape[0]:
        raise ValueError(f'Different no. features for "features" and "result" - {len(features)} and {result.shape[0]} respectively. Check preprocess() and view_roughness() are dropping the same features')

    #Remove unwanted features - standard deviations
    filtered = []
    filtered_names = []
    feature_count = 0
    for i in range(len(features)):
        if 'Std' not in features[i]:
            feature_count += 1
            filtered = np.append(filtered, result[i])
            filtered_names.append(features[i])
    result = np.reshape(filtered, (feature_count, result.shape[1]))
    features = filtered_names
    
    #Get required metric (MAPE or MSE) - MSE stored on odd columns, MAPE stored on even columns
    model_names = model_names[:int(result.shape[1]/2)] #Change the 2 to match number of metrics stored in the data
    filtered = []
    if view_metric == 'MAPE':
        for i in range(result.shape[1]):
            if i%2:
                filtered = np.append(filtered, result[:, i])
    elif view_metric == 'MSE':
        for i in range(result.shape[1]):
            if i%2 == 0:
                filtered = np.append(filtered, result[:, i])
    result = np.reshape(filtered, (-1, result.shape[0]))
    
    # ======= OPTION 1 ======== #
    # Have bars stacked on top of each other
    #Get 14 (len(features)) empty dicts
    empty_dict = []
    for i in range(len(features)):
        empty_dict.append(dict())
    #Populate each dict with each array from result
    #When populating the dict, ensure each value is entered with the corresponding model name that produced the result
    result = np.transpose(result)
    for i in range(len(empty_dict)):
        for j in range(len(result[i, :])):
            empty_dict[i].update({model_names[j]: result[i, j]})
        # sort the new entry into descending order
        empty_dict[i] = dict(sorted(empty_dict[i].items(), key=lambda x:x[1], reverse=True))
    #Make one final dictionary
    final_dict = dict(zip(features, empty_dict))

    #Plot the graph
    #Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html
    width = 0.5  # the width of the bars
    fig, ax = plt.subplots(layout='constrained')

    colour_dict = dict(zip(model_names, mcolors.TABLEAU_COLORS.keys())) # Set bar colours for each model
    
    for feature, dict_ in final_dict.items():
        for model, value in dict_.items():
            p = ax.bar(feature, value, width=width, bottom=0, label=model, color=colour_dict[model])
    
    handles, labels = plt.gca().get_legend_handles_labels() #Put legend into a dictionary to remove duplicates
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    if view_metric == 'MAPE':
        ax.set_title('Comparing MAPE for each output feature for each ML model')
        ax.set_ylabel('Mean Absolute Percentage Error (MAPE)')
    elif view_metric == 'MSE':
        ax.set_title('Comparing MSE for each output feature for each ML model')
        ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_xlabel('Output Features')
    
    plt.show()
    plt.close()
    return 0

    """
    # ======= OPTION 2 ======== #
    # Have bars stacked side by side
    
    #Plot the data
    #Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    data_dict = dict(zip(model_names, result))
    print(data_dict)
    x = np.arange(len(features))
    width = 0.1  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    
    for feature, value in data_dict.items():
        print(feature)
        print(value)
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=feature)
        multiplier += 1

    ax.set_ylabel('Mean Absolute Percentage Error (MAPE)')
    ax.set_title('Comparing MAPE for each output feature for each ML model')
    ax.legend(loc='upper left')
    ax.set_xticks(x + width, features)
    ax.legend(loc='upper left', ncols=len(model_names))
    plt.show()
    plt.close()
    """
    
def view_percent_change_multiple(result, y, output_features):
    #Show percentage change for features for multiple ML models
    #Again multidimensional problem, create subplots where each plot contains the percentage change in value for each sample in a single feature, for every model
    model_names = ["Random Forest", "Gradient Boost", "Linear Regression", "SVR", "KNeighbours", "MLP", "Gaussian Process", "Decision Tree"]
    names = output_features
    if len(names) != result[0].shape[1]:
        raise ValueError(f'Different no. features for "data" and "result" - {data.shape[1]} and {result.shape[0]} respectively. Check preprocess() and view_roughness() are dropping the same features')
    num_cols = 4 #Used to determine number of columns in the subplot
    shape = (int(math.ceil(len(names)/num_cols)), num_cols)
    fig, axs = plt.subplots(shape[0], shape[1], layout='constrained')
    colour_dict = dict(zip(model_names, mcolors.TABLEAU_COLORS.keys())) # Set bar colours for each model
    #Keep using the colour_dict to keep line colours consistent with other functions
    x = np.arange(0, result[0].shape[0], 1) #Should be length of no. samples
    i = 0
    for ax in (axs.flat):
        #i represent features, j represent different models
        for j in range(len(result)):
            data = result[j]
            ax.plot(x,  data[:, i], color=colour_dict[model_names[j]], label=model_names[j])
        ax.set_xlabel('Sample index')
        ax.set_ylabel('Percentage Change')
        ax.set_title(names[i])
        i += 1
        if i == data.shape[1]:
            break
    
    fig.suptitle('Comparing Percentage change for each output feature for each ML model')
    ax.legend()
    plt.show()
    plt.close()

def view_percent_change_single(result, model_name, output_features):
    #Show percentage change for features for a single ML model
    names = output_features
    if len(names) != result.shape[1]:
        raise ValueError(f'Different no. features for "names" and "result" - {len(names)} and {result.shape[1]} respectively. Check preprocess() and view_roughness() are dropping the same features')
    x = np.arange(0, result.shape[0], 1) #Should be length of no. samples
    fig, ax = plt.subplots(layout='constrained')
    for i in range(result.shape[1]):
        ax.plot(x, result[:, i], label=names[i]) #Each line is a different feature this time, so not using colour dict
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Cumulative Percentage Change')
    ax.set_title(f'Cumulative Percentage Change for {model_name}')
    ax.legend()
    plt.show()
    plt.close()
    
    
# =============================== #

# ============ Training / Testing / Optimisation ============ #

def fit_pred_train(clf, X, y, cross_val=False): # Perform fit predict on training data (can use cross_val too)
    if cross_val:
        length = 10 # Number of groups for the cross validation
        MSE_scores = cross_val_score(clf, X, y, cv=length, scoring="neg_mean_squared_error", n_jobs=-1) #neg_mean_absolute_percentage_error
        print(f'Cross Validation MSE: {-MSE_scores.mean()}')
        print(f'Cross Validation std: {MSE_scores.std()}')
        MAPE_scores = cross_val_score(clf, X, y, cv=length, scoring="neg_mean_absolute_percentage_error", n_jobs=-1)
        print(f'Cross Validation MAPE: {-MAPE_scores.mean()}')
        print(f'Cross Validation std: {MAPE_scores.std()}')
        return -MSE_scores.mean(), MSE_scores.std(), -MAPE_scores.mean(), MAPE_scores.std()
    else:
        # Split into train and test splits. Alternative to Cross Validation, not to be used in conjunction.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
        #Fit and predict
        multi_log_regression = clf.fit(X_train, y_train).predict(X_test)
        MSE = mean_squared_error(y_test, multi_log_regression, multioutput='raw_values') #Uncomment raw_values to see MSE for each feature 
        MAPE = mean_absolute_percentage_error(y_test, multi_log_regression, multioutput='raw_values') #Uncomment raw_values to see MAPE for each feature
        print(f'train_test_split MSE: {MSE}')
        print(f'train_test_split MAPE: {MAPE}')
        return MSE, MAPE

def fit_pred_test(clf, X_train, X_test, y_train, y_test): # Fit clf to X_train and test on X_test
    #y_test may potentially be None if using data from generate_test_data
    
    clf = clf.fit(X_train, y_train)
    if y_test == None: #In this case, cannot generate a usual metric, have to generate self-implemented percentage change
        y_pred = clf.predict(X_test)
        per_diff = percentage_change(y_pred)
        return per_diff
    else:
        raise ValueError(f'fit_pred_test not implemented for when y_test != None. Simply add one of the metrics found in fit_pred_train')
        
# =========================================================== #

def main():
    # ============= Options ============= #
    # Choose the ML method and retrieve the model
    method = [0, 0, 0, 0, 0, \
              0, 0, 0]
            #Random Forest, Gradient Boost, Linear Regression, SVR, KNeighbors
            #MLP, Gaussian Process, Decision Tree

    options_features = ['Pressure', 'Speed', 'SRR']
    ranges = [np.arange(1, 3, 0.1), np.arange(200, 2600, 100), np.arange(-0.12, 0.12, 0.01)]
    choice = 2 # Which index is chosen from the above 2 arrays (options_features and ranges)
    cross_val = False

    options = [0,                  0,                0,                 0,                0,                            \
               0,                 0,                               0,                            \
               0,                     0]
    #          0. Single ML model, 1. All ML models, 2. fit_pred_train, 3. fit_pred_test, 4. remove_std_output_features \
    #          5. view_roughness, 6. view_percent_change_multiple, 7. view_percent_change_single \
    #          8. write_back_results, 9. print_results
    options[1], options[2], options[5] = 1, 1, 1
    #Choose which file to get data from - all functions should work with both
    files = ['MLdata.csv', 'MLdata_2.csv']
    csv_file = files[1]

    scale = True
    remove_std = True
    # ============= Option Checking ============== #
    # VALID COMBINATIONS - not implemented ValueError yet
    #from 0: can use (2, 5) or (3, 4, 7) or (3, 7)
    #from 1: can use (2, 5, 8, 9) or (3, 4, 6) or (3, 4, 7). 4 is optional, so are 8 or 9

    # 1, 3, [4], 7 produces the most interesting outputs
    # 2 already removes std columns inside - is not compatible with 4, to change this manually disable it since there is no option for it yet
    # When using 8 or 9 ensure that multioutput='raw_values' is commented out inside fit_pred_train
    # When using 5 ensure multioutput='raw_values' is enabled in fit_pred_train
    # ============= Retrieve data ============= #
    data = import_data(csv_file)
    # Preprocess the data
    X_train, y_train, feature_str = preprocess(data, scaling=scale, remove_std_features=remove_std)
    X_test = generate_test_data(csv_file, options_features[choice], data_range=ranges[choice])
    #X_test returns as dataframe so turn into numpy array if there is no preprocessing
    X_test = X_test.to_numpy()    
    

    # ============= Single method functions ============= #
    if options[0]:
        clf = get_method(method)
        if options[2]:
            result = fit_pred_train(clf, X, y_train, cross_val)
        if options[3]:
            result = fit_pred_test(clf, X_train, X_test, y_train, y_test=None)
        if options[4]:
            result, out_feature_str = remove_std_output_features(result, feature_str) #out_feature_str holds the feature names for the output features
    # ============= Multi method functions ============= #
    if options[1]:
        method = clear_method(method)
        result = []
        for i in range(len(method)):
            method[i] = 1
            clf = get_method(method)
            if options[2]:
                result.append(fit_pred_train(clf, X_train, y_train, cross_val))
            if options[3] and options[4]:
                individual_result = fit_pred_test(clf, X_train, X_test, y_train, y_test=None)
                no_std_result, out_feature_str = remove_std_output_features(individual_result, feature_str)
                result.append(no_std_result)
            elif options[3]:
                result.append(fit_pred_test(clf, X_train, X_test, y_train, y_test=None))
            #optimise_methods(method, X, y_train) # Wouldn't bother with this, takes ages for small amount of gain.
            method[i] = 0
            print("\n")
    
    if not options[4]:
        out_int = int(np.where(feature_str == 'CompRqPost')[0])
        out_feature_str = feature_str[out_int:]
    # ============= Data view functions ============= #
    if options[5]:
        view_roughness(csv_file, result, cross_val, y_train, out_feature_str)
    if options[6]:
        view_percent_change_multiple(result, y_train, out_feature_str)
    if options[0] and options[7]:
        view_percent_change_single(result, str(get_method(method)), out_feature_str)
    elif options[1] and options[7]:
        for i in range(len(method)):
            method[i] = 1
            view_percent_change_single(result[i], str(get_method(method)), out_feature_str)
            method[i] = 0

    # ============= Write back functions ============= #
    if options[1] and (options[8] or options[9]):
        if cross_val:
            result = np.reshape(result, (len(method), 4))
        else:
            result = np.reshape(result, (len(method), 2))
    if options[8]:
        write_back_results(result, cross_val)
    if options[9]:
        print_results(result)
    
    return 0

if __name__ == '__main__':
    main()


#Old code

    """
def optimise_methods(method, X, y): #e.g. clf = optimise_methods(method, clf, TrainBin, y_true_bin)
    # Optimisation is done through grid search cv, which finds the optimal combination of the given options for a particular model
    if method[0]: # Random Forest
        clf = RandomForestRegressor(random_state=0)
        params = [
            {
                'n_estimators': np.arange(100, 1000, 10),
                'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                'max_depth': np.arange(5, 75, 1)
            }
        ]
    elif method[1]: # Gradient Boost
        clf= GradientBoostingRegressor(random_state=0)
        params = [
            {
                'n_estimators': np.arange(100, 1000, 10),
                'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                'criterion': ['friedman_mse', 'squared_error'],
                'max_depth': np.arange(5, 75, 1),
                'learning_rate': [0.0015, 0.002, 0.01]
            }
        ]
    elif method[2]: # Linear Regression
        clf= LinearRegression()
        print('No Optimisation for Linear Regression')
        return clf
    elif method[3]: # SVR
        clf= SVR()
        params = [
            {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': np.arange(0, 10, 1),
                'gamma': ['scale', 'auto'],
                'C': np.logspace(-5, 1, 7)
            }
        ]
    elif method[4]: # KNeighbours
        clf= KNeighborsRegressor()
        params = [
            {

                'n_neighbors': np.arange(0, 11, 1),
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': np.arange(0, 101, 1),
                'p': [1, 2]
            }
        ]
    elif method[5]: # MLP
        clf= MLPRegressor(random_state=0)
        params = [
            {
                'hidden_layer_sizes': np.arange(100, 1000, 200),
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'max_iter': [1000]
            }
        ]
    elif method[6]: # Gaussian Process
        clf= GaussianProcessRegressor(random_state=0)
        print('No Optimisation for Gaussian Process')
        return clf
    elif method[7]: # Decision Tree
        clf= DecisionTreeRegressor(random_state=0)
        params = [
            {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'splitter': ['best', 'random'],
                'max_depth': np.arange(5, 50, 5)
            }
        ]
    else:
        raise ValueError("No ML method selected")
    gs_clf = GridSearchCV(clf, param_grid=params, scoring='neg_mean_squared_error', n_jobs=-1)
    gs_clf = MultiOutputRegressor(gs_clf)
    gs_clf.fit(X, y)
    print(gs_clf.best_params_)
    print(gs_clf.score(X, y))
"""
