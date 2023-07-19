import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
"""
To implement:
Cross validation or train test split
"""
# ============ Utils ============ #

def import_data(csv_file):
    return pd.read_csv(csv_file)

def split(data):
    " Split the data into X and y. "
    # No Scaling
    end_inp = int(np.where(data.columns == 'CompRqPost')[0]) #Find the index in data headers where input ends and output begins
    input_data = data.iloc[:, 0:end_inp] # X
    output_data = data.iloc[:, end_inp:] # y
    

    """ #Scale only the input
    inp_columns = input_data.columns
    out_columns = output_data.columns
    
    input_data = StandardScaler().fit_transform(X=input_data, y=output_data) #Cannot do normal scaler here since it will change -1 values on CyclesToFF
    input_data = pd.DataFrame(data=input_data, columns=inp_columns)
    output_data = pd.DataFrame(data=output_data, columns=out_columns)
    """
    
    """ #Scale the input and output, then replace the CyclesToFF values manually with -1

    #Firstly find the rows in CyclesToFF which have -1
    minus_1 = np.where(data.loc[:, 'CyclesToFF'].to_numpy() == -1)[0]

    #Scale the data
    column_names = data.columns.to_list()
    data = StandardScaler().fit_transform(data)
    data = pd.DataFrame(data, columns=column_names)

    #Replace the original values with -1
    for value in minus_1:
        data.loc[value, 'CyclesToFF'] = -1

    #Split the data
    end_inp = int(np.where(data.columns == 'CompRqPost')[0]) #Find the index in data headers where input ends and output begins
    input_data = data.iloc[:, 0:end_inp] # X
    output_data = data.iloc[:, end_inp:] # y
    """
    
    return input_data, output_data
    

def get_method(method, regression=True):
    if regression:
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
        elif method[5]: # Gaussian Process
            clf= GaussianProcessRegressor(random_state=0)
        elif method[6]: # Decision Tree
            clf= DecisionTreeRegressor(random_state=0)
        else:
            raise ValueError("No regressor method selected")
        print(clf)
        return MultiOutputRegressor(clf)
    else: #Classification models
        if method[0]:  # Did not optimise, terrible results so not worth the time
            clf = RandomForestClassifier(random_state=0)
        elif method[1]:  # Not optimisable
            clf = ExtraTreesClassifier(random_state=0)
        else:
            raise ValueError("No classifier method selected")
        print(clf)
        return clf

def write_back_results(acc, mse, regressors, classifiers, dest_file):
    # Write back 'result' to 'dest_file'.
    # 'result' should be a Nx2 matrix for N number of models.
    result = np.hstack((acc, mse))
    result = np.reshape(result, (-1, 2))
    #Rename each half of the keys so they are all unique (otherwise they get overwritten if both have the same names for regressors and classifiers)
    regressors = np.hstack(([name + ' Accuracy' for name in regressors], [name + ' MSE' for name in regressors]))
    print(regressors)
    print(result)
    result_dict = dict(zip(regressors, result))
    # Convert numpy array to df, attach headers then write to csv
    
    df = pd.DataFrame.from_dict(result_dict, orient='index', columns=classifiers)
    df.to_csv(dest_file) # Write to the csv file at location 'dest_file'
    return None

def print_results(result):
    # Show results as a box and whisker plot
    model_names = ["Random Forest", "Gradient Boost", "Linear Regression", "SVR", "KNeighbours", "MLP", "Gaussian Process", "Decision Tree"]
    to_del = [5, 6]
    result = np.delete(result, to_del, 0)
    model_names = np.delete(model_names, to_del, 0)
    result = np.reshape(result[:, 0], (1, len(result[:, 0])))
    """ #Code for whiskers, doesn't work since whisker min / maxes cannot be specified
    result_0 = np.reshape(result[:, 0], (1, len(result[:, 0])))
    result_1 = np.reshape(result[:, 1], (1, len(result[:, 1])))/2
    whisks = np.vstack((np.add(result_0, result_1), np.subtract(result_0, result_1))) #top row max, bottom row mins
    #plt.boxplot(result_0, whis=whisks)#, usermedians=result_0)
    """
    
    plt.boxplot(result, labels=model_names)
    plt.title('Multioutput Regression Results')
    plt.show()
    plt.close()
    return None


# =============================== #

# ============ Training / Testing / Optimisation ============ #

def test_split_tests(clf_r, clf_c, X, y):
    # Code used to produce the test_size vs accuracy graph
    accuracies = []
    split_range = np.arange(0.1, 0.9, 0.05)
    for i in split_range:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=0)
        accuracies.append(classification(clf_c, X_train, X_test, y_train, y_test, return_acc=True))
    x = np.arange(0, len(accuracies), 1)
    plt.plot(split_range, accuracies, 'b-')
    plt.title(f'Testing effect of number of test values on accuracy ({clf_c})')
    plt.xlabel('Percentage of test values')
    plt.ylabel('Accuracy')
    plt.show()
    plt.close()
    return 0

def fit_pred(clf_r, clf_c, X, y):
    # Split into train and test splits.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0) #Also shuffles the data
    y_pred, accuracy = classification(clf_c, X_train, X_test, y_train, y_test)

    #Create dictionary to convert between indexes of dataframe and numpy array
    converter = dict(zip(np.arange(0, len(y_pred), 1), y_test.index.to_list()))

    #Find which rows are predicted as -1 in the dataframe y_pred
    minus_1 = np.where(y_pred == -1)[0]
    converted = []
    for item in minus_1:
        converted.append(converter[item])
    #Remove rows predicted as -1 from y_test and X_test
    #Leave any other -1's as they are so the error will show
    y_test = y_test.drop(converted, axis=0)
    X_test = X_test.drop(converted, axis=0)

    #Print number of -1s still left in the test array
    y_list = y_test.loc[:, 'CyclesToFF'].to_numpy()
    print(f'Number of -1s still in the array: {len(np.where(y_list == -1)[0])}')
    
    MSE = regression(clf_r, X_train, X_test, y_train, y_test)
    return accuracy, MSE

def classification(clf, X_train, X_test, y_train, y_test, return_acc=False):
    #Remove all output features apart from CyclesToFF 
    y_train = y_train.loc[:, 'CyclesToFF']
    y_test = y_test.loc[:, 'CyclesToFF']

    # For Classification, any values not == -1 set to 1
    y_train = y_train.where(y_train == -1, 1)
    y_test = y_test.where(y_test == -1, 1)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f'Classification accuracy: {accuracy}')
    if return_acc:
        return accuracy
    else:
        #Print the number of incorrectly predicted values
        print(f'Number of incorrectly classified rows: {len(np.where(y_pred != y_test)[0])}')
        return y_pred, accuracy

def regression(clf, X_train, X_test, y_train, y_test):
    #Fit and predict
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    multioutput = True
    if multioutput:
        column_names = y_test.columns.to_list()
        MSE = mean_squared_error(y_test, y_pred, multioutput='raw_values') #Shows MSE for each feature
        """ # Works but cannot read the labels when there are too many of them
        plt.boxplot(np.reshape(MSE, (1, len(MSE))), labels=y_test.columns.to_list())
        plt.show()
        plt.close()
        """
        write_back_output(y_pred, column_names)
        col_index = np.where(MSE >= 1)[0]
        col_names = y_test.columns.to_numpy()[col_index]
        print(f'Output features that have a loss >= 1: {col_names}')
        print(f'Their respective individual losses: {MSE[col_index]}')
        
        print('\nThe following shows the predicted vs the actual values of the above columns:\n')
        for val in col_index:
            print(y_test.columns.to_numpy()[val])
            print('Predicted | Actual')
            for j in range(len(y_pred[:, val])):
                print(f'{round(y_pred[j, val], 1)} | {round(y_test.to_numpy()[j, val], 1)}')
            print('\n')
    else:
        MSE = mean_squared_error(y_test, y_pred)
        print(f'Regression MSE: {MSE}')
    return MSE

def write_back_output(result, column_names):
    print(result)
    dest_file = 'output_y_pred.csv'
    df = pd.DataFrame(result, columns=column_names)
    df.to_csv(dest_file) # Write to the csv file at location 'dest_file'
    
    
    

# =========================================================== #

def main():
    #return 0
    # Choose the ML method and retrieve the model
    method_regr = [0, 1, 0, 0, 0, \
              0, 0]
    regr_names = ['Random Forest', 'Gradient Boost', 'Linear Regression', 'SVR', 'KNeighbors', \
                  'Gaussian Process' , 'Decision Tree']
    method_classi = [0, 1]
    classi_names = ['Random Forest', 'Extra Trees']
    
    # Retrieve the data
    csv_file = 'MLdata.csv'
    data = import_data(csv_file)
    
    # Split up the data into input and output columns - multiple outputs are used here
    X, y = split(data)
    
    clf_r = get_method(method_regr)
    clf_c = get_method(method_classi, regression=False)
    #test_split_tests(clf_r, clf_c, X, y)
    acc, mse = fit_pred(clf_r, clf_c, X, y)
    return 0
    
    accuracies = []
    MSEs = []
    for i in range(len(method_regr)):
        method_regr[i] = 1
        clf_r = get_method(method_regr)
        for j in range(len(method_classi)):
            method_classi[j] = 1
            clf_c = get_method(method_classi, regression=False)
    
            acc, mse = fit_pred(clf_r, clf_c, X, y)
            accuracies.append(acc)
            MSEs.append(mse)
            method_classi[j] = 0
            print("\n")
        method_regr[i] = 0
    #write_back_results only works when multioutput = False in function regression().
    #write_back_results(accuracies, MSEs, regr_names, classi_names, dest_file='classifier to regressor with all columns.csv')
    return 0

if __name__ == '__main__':
    main()

