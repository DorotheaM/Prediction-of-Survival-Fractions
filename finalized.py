import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.impute import KNNImputer
import itertools
from itertools import accumulate, chain, repeat, tee
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from IPython.display import Image
from sklearn import tree
from sklearn.tree import export_graphviz
import pydotplus
import joblib

def chunk(xs, n):
    """
    Break a list xs into n equal sublists
    """
    assert n > 0
    L = len(xs)
    s, r = divmod(L, n)
    widths = chain(repeat(s+1, r), repeat(s, n-r))
    offsets = accumulate(chain((0,), widths))
    b, e = tee(offsets)
    next(e)
    return [xs[s] for s in map(slice, b, e)]

def performance(y_test, preds, model, model_st, headers):
    """
    Get predictions and true values and scatter plot them. Additionally subplot a linear regression line on the points
    and get its equation y = a *x + b. The closer the line to the y = x line the better the prediction.
    Additionally break the preds and truth paired set into 4 equal chunks and plot the mean and variance in order to visualize
    any trend to the error of the prediction. Additionally plot the feature importances of the model.
    """
    model_name = str(model).split('(')[0]
    mse = mean_squared_error(y_test,preds)
    r2 = r2_score(y_test,preds)

    #Make linear regression line on points (pred,truth)
    linear_regressor = LinearRegression()
    linear_regressor.fit(y_test.reshape(-1, 1),np.array(preds).reshape(-1, 1))
    y_linear = linear_regressor.predict(y_test.reshape(-1, 1))
    a = str(round(linear_regressor.coef_[0][0],4))     #alpha coefficient of linear regression
    b = round(linear_regressor.intercept_[0],4)     #beta intercept of linear regression
    score = lambda i: ("+" if i > 0 else "") + str(i)     #get the sign of the intercept
    b = score(b)

    #Linear regression line
    reg_line = plt.figure()
    plt.style.use('ggplot')
    plt.plot(y_test, y_linear, 'black', label='y='+a+'x'+b)     #plot reg line
    plt.scatter(y_test, preds, alpha=0.5, color='xkcd:electric blue')     #scatter plot truth vs pred
    # plt.xlim(min_,max_); plt.ylim(min_,max_)
    tfont = {'fontname':'Times New Roman'}
    plt.xlabel('Test_y', labelpad=3, **tfont)
    plt.ylabel('Predictions', labelpad=3, **tfont)
    plt.title('Test_y VS Predictions for '+model_st+' '+model_name, **tfont)
    plt.legend(title='MSE '+str(round(mse,4))+'\n'+'$R^2$: '+str(round(r2,4)), fontsize='small')
    plt.show()
    plt.close()

    fig = plt.figure()
    gs = gridspec.GridSpec(2,4,figure=fig)
    ax=[None]*4
    means=[]
    vars_=[]

    #Create truth and pred pairs in chunks
    y_test_sorted, preds_sorted = zip(*sorted(zip(y_test, preds)))
    zipped = list(zip(y_test_sorted, preds_sorted))
    lists = chunk(zipped,4)
    
    #Calculate mean and variance for every chunk
    for list_,index in zip(lists,range(4)):
        res = [truth - pred for truth, pred in list_]     #residual is the difference between the actual and the predicted value
        mean = np.mean(res)     #mean of residuals for each chunk
        var = np.var(res)     #variance of residuals
        means.append(mean)
        vars_.append(var)
        
        ax[index] = fig.add_subplot(gs[1, index])   #create boxes
        ax[index].hist(res, bins=40, label='mean: '+str(round(mean,4)) + '\n' + 'variance: '+str(round(var,4)), color='xkcd:peach')     #distribution of residuals
        ax[index].set_title('slice: '+str(index +1), fontdict=dict(fontsize=10), **tfont)
        ax[index].legend(fontsize = 'x-small')
    second =  fig.add_subplot(gs[0, :]) 
    second.plot(means, color='xkcd:pale purple',  marker='.', label='mean')
    second.plot(vars_ , color='xkcd:sky',  marker='.', label='variance')
    second.set_title('Truth - Pred Mean & Variance for '+model_st+' '+model_name, **tfont)
    second.legend()
    plt.show()
    plt.close()

    #Feature Importances
    importances = list(model.feature_importances_)
    model_importances = [(feature, round(importance, 2)) for feature, importance in zip(headers_list, importances)]
    model_importances = sorted(model_importances, key = lambda x: x[1], reverse = True)
    print('Feature Importances for '+model_st+' '+model_name)
    [print('Variable: {:25} Importance: {}'.format(*pair)) for pair in model_importances]  
    print()
    
    #Figure of Importances
    x_values_model = list(range(len(importances)))    #list of x locations for plotting
    plt.figure(figsize=(13,9))
    plt.barh(x_values_model, sorted(importances, reverse=True), color='xkcd:cool green')     #makes a horizontal bar chart
    plt.yticks(x_values_model, [x for x,y in model_importances], rotation='horizontal')
    plt.xlabel('Importance', **tfont); plt.ylabel('Variable', labelpad=3, **tfont); plt.title('Variable Importances for '+model_st+' '+model_name, **tfont);
    plt.show()
    
def evaluate(model, model_st, train_X, train_y, test_X, test_y):

    model_name = str(model).split('(')[0]
    model_string = model_st.split('_')[0]
    predictions_test = model.predict(test_X)
    predictions_train = model.predict(train_X)
    abs_error_test = abs(predictions_test - test_y)   #calculate the absolute errors for the test set
    abs_error_train = abs(predictions_train - train_y)   #calculate the absolute errors for the train set
    error_test =  test_y - predictions_test
    error_train = train_y - predictions_train
    mae = np.mean(abs_error_test)
    mape = np.mean(100*(abs_error_test/test_y))
    print('Performance of', model_string, model_name,':')
    #Evaluation Metrics
    print('Mean Squared Error of Train data: {:0.4f}' .format(mean_squared_error(train_y,predictions_train)))
    print('Mean Squared Error of Test data: {:0.4f}' .format(mean_squared_error(test_y,predictions_test)))
    print('Root Mean Squared Error of Train data: {:0.4f}'.format(np.sqrt(mean_squared_error(train_y,predictions_train))))
    print('Root Mean Squared Error of Test data: {:0.4f}' .format(np.sqrt(mean_squared_error(test_y,predictions_test))))
    print('Mean Absolute Error: {:0.4f}' .format(mae), 'degrees')
    print('Mean Absolute Percentage Error: {:0.4f}%' .format(mape))
    print('Normalized Root Mean Squared Error of Train data: {:0.4f}' .format(np.sqrt(mean_squared_error(train_y,predictions_train))/(train_y.max() - train_y.min())))
    print('Normalized Root Mean Squared Error of Test data: {:0.4f}' .format(np.sqrt(mean_squared_error(test_y,predictions_test))/(test_y.max() - test_y.min())))
    print('Variance Score - R2: {:0.4f}' .format(model.score(test_X,test_y)))
    print()
    
    #Plot for residual error and overfitting
    over = plt.figure()
    plt.style.use('ggplot')
    plt.scatter(train_y, error_train, color='xkcd:strawberry', alpha=0.7, s=19, label='Train data')
    plt.scatter(test_y, error_test, color='xkcd:golden', alpha=0.7, s=19, label='Test data')
    plt.axhline(color='black')
    tfont = {'fontname':'Times New Roman'}
    plt.xlabel('Y values', labelpad=2, **tfont)
    plt.ylabel('Error values', labelpad=2, **tfont)
    plt.title('Residual Errors for '+model_string+' '+model_name, **tfont)
    plt.legend()
    plt.show()

    return mape

def save_params(grid):
    """
    save best parameters from grid search
    """
    # model_name = str(model).split('(')[0]
    param = grid.best_params_   
    param_name = 'Parameters of '+model_name
    print(param_name,'\n',param,'\n')

    return param

#Creation of DataFrame and adjustments to the rows and cols
data = pd.read_excel('ijms-568854-supplementary_new-1-2.xlsx', sheet_name='SD1', header=1)
data = pd.melt(data, id_vars=['Cell Line', 'PMID', 'Cell Cycle', 'Timing of Cell Seeding', 
                         'Radiation Type', 'Photon Radiation', 'Photon Rad (MeV)', 'Dose Rate (Gy/min)'], 
                         value_vars=['SF2', 'SF4', 'SF6','SF8'], var_name='SFs', value_name='SF values')    #convert SF cols to rows 

data['Dose (Gy)'] = ""     #add new column
data = data[data['SF values'].notna()]     #keep the rows whose SF isn't null
data['Dose (Gy)'] = data['SFs'].astype(str).str[2].astype(int)     #fill in the Dose col

#Selection of features
features =  ['Cell Line', 'Cell Cycle', 'Timing of Cell Seeding', 'Radiation Type',
             'Photon Rad (MeV)', 'Dose Rate (Gy/min)', 'SF values', 'Dose (Gy)']
data_final = data[features]

## Introductory Plots
# pp = sns.pairplot(data_final, vars=['Photon Rad (MeV)', 'Dose Rate (Gy/min)', 'Dose (Gy)', 'SF values'], diag_kind='kde', diag_kws=dict(color='orange'), plot_kws=dict(s=12, color='brown'))
# pp.savefig(fname='Pairplot of numerical columns (kde)(size=12)')
# plt.scatter(data_final['Photon Rad (MeV)'], data_final['SF values'], alpha=0.5, c='cyan');  plt.xlabel('Photon Rad (MeV)');  plt.ylabel('SF values');  plt.show()
# plt.scatter(data_final['Dose Rate (Gy/min)'], data_final['SF values'], alpha=0.5, c='green');  plt.xlabel('Dose Rate (Gy/min)'); plt.ylabel('SF values');  plt.show()
# plt.scatter(data_final['Dose (Gy)'], data_final['SF values'], alpha=0.5, c='purple');  plt.xlabel('Dose (Gy)'); plt.ylabel('SF values');  plt.show()
# plt.hist(data_final['Cell Line'], color='yellow');  plt.xlabel('Cell Line');  plt.ylabel('Counts');  plt.show()
# cl = sns.distplot(data_final['Cell Line']); fig = cl.get_figure();  fig.savefig('Cell Lines versus SF values')
# cat = plt.figure()
# plt.style.use('ggplot')
# plt.subplot(1,2,1)
# plt.scatter(data['Cell Line'], data['SF values'], alpha=0.5, s= 9, c='xkcd:mauve')     #scatter plot truth vs pred
# tfont = {'fontname':'Times New Roman'}
# plt.xlabel('Cell Line', labelpad=3, **tfont); plt.ylabel('SF values', labelpad=3, **tfont)
# plt.subplot(1,2,2)
# plt.bar(data['Cell Line'], data['SF values'], alpha=0.01, color='xkcd:mauve')
# plt.xlabel('Cell Line', labelpad=3, **tfont); plt.ylabel('SF values', labelpad=3, **tfont)
# cat.suptitle('Relationship between Cell Line and Target Value', **tfont)
# plt.show(); # plt.close()

#Correlation
print('Pearson: \n', data_final.corr(method = 'pearson')); print()

#Manipulate categorical values
encoder = ce.one_hot.OneHotEncoder()
data_final = encoder.fit_transform(data_final) 
data_final.to_excel("encoded.xlsx")     #convert to an excel file in order to visualize
data_final_wt = data_final.drop('SF values', axis = 1)     #drop the target value
headers = data_final_wt.columns.values     #headers' names without the target value
headers_list = list(data_final_wt.columns)     #headers' names without the target value as a list

#Fill in the missing values
imputer = KNNImputer(n_neighbors=2)
imputed_dataset = imputer.fit_transform(data_final)

#Features and Target as arrays respectively
X = np.delete(imputed_dataset, 18, axis=1)
y = np.array(imputed_dataset[:,18])

#Train-Test Split
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25)

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

models = [RandomForestRegressor(), GradientBoostingRegressor()] 

#Create the parameter grid for each model to search for optimal parameters
param_grid_rf = {
    'criterion'        : ['mse', 'mae'],
    'max_depth'        : [None, 3, 4],
    'min_samples_leaf' : [1, 2, 3],
    'min_samples_split': [2, 5, 10],
    'n_estimators'     : [100, 500, 1000]
                }   #162*5=810 combos 25-30 mins
param_grid_gb = {
    'criterion'        : ['mse', 'mae', 'friedman_mse'],
    'learning_rate'    : [0.01, 0.05, 0.09],
    'max_depth'        : [None, 3, 6],
    'min_samples_leaf' : [1, 2, 3],
    'min_samples_split': [2, 5, 10],
    'n_estimators'     : [100, 500, 1000],
    'subsample'        : [0.9, 1.0]
                }   #1458*5=7290 combos apprx 6 hrs

grids = [param_grid_rf, param_grid_gb]

#Instantiate the grid search model, Save scores, Evaluate model with best parameters
for model, grid in zip(models,grids):
    base_model = model   #using the default parameters provided by scikit-learn
    base_model.fit(train_X, train_y)  
    base_mape = evaluate(base_model, 'base_model', train_X, train_y, test_X, test_y)   #evaluate base model using all the aforementioned metrics
    
    grid_search = GridSearchCV(estimator = model, param_grid = grid, cv = 5, n_jobs = -1, verbose = 1)   #using the combination of parameters given as a result of the grid search
    grid_search.fit(train_X, train_y)

    print(grid_search,'\n')
    print(grid_search.best_estimator_,'\n')
    print(grid_search.best_score_,'\n')
    print(grid_search.best_params_,'\n')

    cv_results = grid_search.cv_results_
    model_name = str(model).split('(')[0]
    scores = pd.DataFrame(cv_results).to_excel('CV_Results_'+model_name+'.xlsx')   #save all scores from the grid search to an excel file

    best_grid = grid_search.best_estimator_
    grid_mape = evaluate(best_grid, 'best_model', train_X, train_y, test_X, test_y)   #evaluate best model using all the aforementioned metrics

    print('Improvement of {:0.4f}%.'.format( 100 * (grid_mape - base_mape) / base_mape),'\n')   #improvement of mape

    base_params = {}
    parameters = save_params(grid_search)
    params = [base_params, parameters]

    for param in params:
        model.set_params(**param)
        model.fit(train_X,train_y)
        predictions = model.predict(test_X)
        if param == base_params:
            performance(test_y, predictions, model, 'base', headers)   #study the performance of base model through diagrams and relations between actual and predicted values
        elif param == parameters:
            performance(test_y, predictions, model, 'best', headers)   #study the performance of best model through diagrams and relations between actual and predicted values
