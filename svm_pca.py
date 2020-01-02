import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import time
import seaborn as sns
import sys
import warnings
np.set_printoptions(threshold=sys.maxsize)
from sklearn.preprocessing import MinMaxScaler


# - - - prepreprocess - - -
def encode_categorical_features(dataframe):
    #one hot encoding of different cultivars, delete the original
    dataframe = pd.concat([dataframe, pd.get_dummies(dataframe["cultivar"], prefix="cultivar")], axis=1)
    del dataframe["cultivar"]

    #binary encoding of field location
    dataframe = dataframe.replace("California", 0)
    dataframe = dataframe.replace("Arkansas", 1)

    del dataframe["compartment"]
    del dataframe["year"]

    return dataframe



# - - - microbiome tranformation - - -
def log2_1000_transform_train(dataframe):

    metadata = dataframe.iloc[:,:10]
    dataframe.drop(dataframe.columns[:10], axis=1, inplace=True)
    X = dataframe.to_numpy()

    for i in range(0,len(X)):
        row = X[i]
        suma = sum(row)
        for index in range(0,len(row)):
            row[index] = math.log2((row[index]/suma * 1000)+0.00001)

    ret_val = pd.DataFrame(X, columns= dataframe.columns)
    ret_val = pd.concat([metadata,ret_val], axis=1)
    return ret_val

def log2_1000_transform_test(dataframe):

    metadata = dataframe.iloc[:,:9]
    dataframe.drop(dataframe.columns[:9], axis=1, inplace=True)
    X = dataframe.to_numpy()

    for i in range(0,len(X)):
        row = X[i]
        suma = sum(row)
        for index in range(0,len(row)):
            row[index] = math.log2((row[index]/suma * 1000)+0.00001)

    ret_val = pd.DataFrame(X, columns= dataframe.columns)
    ret_val = pd.concat([metadata,ret_val], axis=1)
    return ret_val

# - - - utilities - - -
def join_taxa(metadata_frame, out_frame):
    ret_val = metadata_frame.join(out_frame.set_index('sample_name'), on='sample_name')
    return ret_val


def extract_cv_categories(dataframe):
    cv_list = dataframe["cv_category"]
    del dataframe["cv_category"]
    return cv_list

def extract_ml_format_data(dataframe):
    Y = dataframe["age"]
    X = dataframe.iloc[:, 2:]
    X = X.to_numpy()
    Y = Y.to_numpy()

    stat_labels = dataframe.iloc[:, 0:2] # a dataframe

    return X,Y, stat_labels


def remove_low_otus(train_joined, test_joined):
    temp = train_joined.iloc[:,10:]
    condition_dataframe = (temp.sum() <= 50)
    columns_to_remove = []
    for item in condition_dataframe.iteritems():
        if item[1] == True:
            columns_to_remove.append(item[0])
    for column in columns_to_remove:
        del test_joined[column]
        del train_joined[column]

# - - - ml algorithms - - -
def consturct_final_model(X, Y, cv_categories, removal_step, min_features, param_dictionary, score_collector):

#VISALIZATION OF CV SPLITS:
#                         cv1_train                           cv1_test                     test
# ----------------------------------------------- // ---------------------- /// -------------------

    #create a new dummie rf model just to exploit GridSearchCV's functionality to generate all param combinations
    dummie_estimator = SVR()
    searcher = GridSearchCV(dummie_estimator, param_grid=param_dictionary, cv=2)
    searcher.fit(X, Y)

    # algorithm
    best_score = 10000  # init start values
    ret_val = None  # whole pipeline
    best_param_feature_reduction_scores = []
    for param_selection_dictionary in searcher.cv_results_["params"]:
        estimator = SVR(**param_selection_dictionary)  # unpacks the dictionary and passes the values
        pca = PCA()
        pipeline_parameters = {'pca__n_components': range(min([X.shape[1], round(X.shape[0]*0.8)]),min_features,-removal_step)}
        pipeline = Pipeline([('pca', pca), ('estimator', estimator)])

        grid_search = GridSearchCV(pipeline, param_grid=pipeline_parameters,cv=StratifiedKFold(n_splits=6, shuffle=True).split(X, cv_categories), n_jobs = -1,scoring="neg_mean_squared_error")
        grid_search.fit(X,Y)


        score_collector.append(-grid_search.cv_results_["mean_test_score"])
        if -grid_search.best_score_ < best_score:
            best_score = -grid_search.best_score_
            ret_val = grid_search.best_estimator_
            best_param_feature_reduction_scores = -grid_search.cv_results_["mean_test_score"]


    score_collector = np.vstack(score_collector)
    return ret_val, score_collector, best_param_feature_reduction_scores

def final_fit_predict(X_train,Y_train, X_test, Y_test, pipeline):

    Y_predictions_train = pipeline.predict(X_train)
    Y_predictions_test = pipeline.predict(X_test)

    mse_train = mean_squared_error(Y_train, Y_predictions_train)
    mse_test = mean_squared_error(Y_test, Y_predictions_test)

    return (mse_train, mse_test, Y_predictions_train, Y_predictions_test)


def final_predict(X_train,Y_train, X_test, Y_test, selector):
    estimator = selector.estimator_
    feature_indices = np.asarray(np.where(selector.support_ == True))
    feature_indices = feature_indices[0]
    X_test_reduced = X_test[:, feature_indices]
    X_train_reduced = X_train[:, feature_indices]

    Y_predictions_test = estimator.predict(X_test_reduced)
    Y_predictions_train = estimator.predict(X_train_reduced)

    mse_test = mean_squared_error(Y_test, Y_predictions_test)
    mse_train = mean_squared_error(Y_train, Y_predictions_train)

    return (mse_train, mse_test)


def select_new_params(param_dictionary):

    n_estimators = param_dictionary["n_estimators"]
    max_depth = str(param_dictionary["max_depth"])
    max_features = [str(param_dictionary["max_features"])]
    min_samples_split = param_dictionary["min_samples_split"]
    min_samples_leaf = param_dictionary["min_samples_leaf"]
    bootstrap = [param_dictionary["bootstrap"]]

    # has to be > 1 soooo...
    if min_samples_split == 2:
        min_samples_split = 3
    if min_samples_leaf == 2:
        min_samples_leaf = 3

    new_param_dictionary = {
        "n_estimators": [n_estimators-50, n_estimators, n_estimators+50],
            "max_features": max_features,
        "min_samples_split": [min_samples_split-1, min_samples_split, min_samples_split+1],
        "min_samples_leaf": [min_samples_leaf-1, min_samples_split, min_samples_split+1],
        "bootstrap": bootstrap
    }
    return new_param_dictionary

# - - - outputs - - -
def print_feature_selection_stats(X_train, rfecv):

    print("Optimal number of features: %d \nCV1 MSE: %.2f" % (rfecv.n_features_, abs(np.max(rfecv.n_features_)) ) )
    print(" Mask of the features: " + str(rfecv.support_.shape))
    print(rfecv.support_)

    print("Rakinf of the features: " + str(rfecv.ranking_.shape))
    print(rfecv.ranking_)

    print("Scorees shape:" + str(rfecv.grid_scores_.shape))
    print(rfecv.grid_scores_)

    print("X shape: " + str(X_train.shape))


def plot_feature_selection(X_train, best_param_feature_reduction_scores ,file_name, removal_step, min_features):
    w = 7
    h = 5
    d = 70
    plt.figure(figsize=(w, h), dpi=d)
    plt.xlabel("Number of features")
    plt.ylabel("MSE CV1")
    X_plot = np.arange(min([X_train.shape[1], round(X_train.shape[0]*0.8)]), min_features, -removal_step)
    Y_plot = best_param_feature_reduction_scores

    min_value = min(best_param_feature_reduction_scores)
    n = X_plot[np.argmin(best_param_feature_reduction_scores)]
    plt.suptitle(("Optimal feature cnt: %d; CV MSE: %.2f" % (n, min_value)), fontsize=12, y=0.75)
    plt.plot(X_plot,Y_plot)
    plt.xlim(min([X_train.shape[1], round(X_train.shape[0]*0.8)]),0)
    plt.grid()
    plt.axvline(x=n, color="y")
    plt.savefig(file_name)
    plt.show()


def plot_mean_sd(X_train, score_collector, file_name, removal_step, min_features):
    X_plot = np.arange(min([X_train.shape[1], round(X_train.shape[0]*0.8)]), min_features, -removal_step)

    df = pd.DataFrame(score_collector, columns=np.array(X_plot))
    ax = sns.tsplot(data=df.values, time=X_plot)

    mean = df.mean(axis=0).to_numpy()
    std = df.std(axis=0).to_numpy()
    ax.errorbar(X_plot, mean, yerr=std, fmt='-o')  # fmt=None to plot bars only
    ax.invert_xaxis()
    plt.xlim(min([X_train.shape[1], round(X_train.shape[0]*0.8)]), 0)
    ax.set(xlabel="Feature count", ylabel="MSE CV")
    plt.savefig(file_name)
    plt.show()




def write_stats_file(mse_train, mse_test, pipeline, best_param_feature_reduction_scores,file_name, dataframe, execution_time):

    f = open(file_name, "w")
    content = "Test MSE: %.2f \n" % mse_test
    content = content + "Train MSE: %.2f \n" % mse_train
    content = content + "Optimal number of features: %d. Minimal MSE in CV: %.2f \n" % (pipeline["pca"].n_components_, min(best_param_feature_reduction_scores))
    content = content + "Time to find the model params: " + str(execution_time) + " seconds."
    content = content + "Chosen model params:\n"
    content = content + str(pipeline["estimator"].get_params()) + "\n"

    f.write(content)
    print(content)
    f.close()


def write_predictions(file_name, dataframe, predictions_array):
    predictions_array = np.multiply(np.ones(len(predictions_array)), predictions_array)

    dataframe["age_predicted"] = predictions_array
    dataframe.to_csv(file_name, sep='\t', index=False)




# * * * * * * * *
# - - - MAIN - - -
# * * * * * * * *
endosphere_train = "data/endosphere/train.tsv"
endosphere_test = "data/endosphere/test.tsv"

rizosphere_train = "data/rizosphere/train.tsv"
rizosphere_test = "data/rizosphere/test.tsv"

compartment = "rizosphere"
level_file = "phylum.tsv"
method_location = "svm/pca/"

otu_file = "data/otu_tables/" + level_file
outputs_location = "outputs/" + method_location

# load data into pandas dataframe
endosphere_train_data = pd.read_csv(endosphere_train,delimiter='\t',encoding='utf-8')
endosphere_test_data = pd.read_csv(endosphere_test,delimiter='\t',encoding='utf-8')
rizosphere_train_data = pd.read_csv(rizosphere_train,delimiter='\t',encoding='utf-8')
rizosphere_test_data = pd.read_csv(rizosphere_test,delimiter='\t',encoding='utf-8')
otu_frame = pd.read_csv(otu_file,delimiter='\t',encoding='utf-8')

if(compartment == "rizosphere"):
    train_set = rizosphere_train_data
    test_set = rizosphere_test_data
else:
    train_set = endosphere_train_data
    test_set = endosphere_test_data


#pre preprocess
train_set_categorized_encoded = encode_categorical_features(train_set)
test_set_encoded = encode_categorical_features(test_set)


#join microbiome data with metadata
train_joined = join_taxa(train_set_categorized_encoded, otu_frame)
test_joined = join_taxa(test_set_encoded, otu_frame)
# the extra column of the train dataframe is used to preserve category balance in cross validation
print("Train dataframe shape: " + str(train_joined.shape))
print("Test dataframe shape: " + str(test_joined.shape))


remove_low_otus(train_joined, test_joined)
print("No variance OTUS removed train dataframe shape: " + str(train_joined.shape))
print("No variance OTUS removed test dataframe shape: " + str(test_joined.shape))

# transform microbiome data
train_joined = log2_1000_transform_train(train_joined)
test_joined = log2_1000_transform_test(test_joined)


cv_categories = extract_cv_categories(train_joined)

(X_train,Y_train, stat_labels_train) = extract_ml_format_data(train_joined)
(X_test,Y_test, stat_labels_test) = extract_ml_format_data(test_joined)

scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)

removal_step = 5
min_features = 5
start_time = time.time()

initial_param_dictionary = {
                            "kernel": ["poly","rbf"],
                            "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            "epsilon": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                            "gamma" : ["scale","auto"]
                           }
# do a two turn finding of parameters
score_collector = []
(pipeline, score_collector, best_param_feature_reduction_scores) = consturct_final_model(X_train,Y_train, cv_categories, removal_step=removal_step, min_features=min_features, param_dictionary=initial_param_dictionary, score_collector=score_collector)
end_time = time.time()

# output results
file_name = outputs_location + compartment + "_" + level_file[0:-4]+ "_step" + str(removal_step)
plot_feature_selection(X_train, best_param_feature_reduction_scores, file_name+".png", removal_step=removal_step, min_features=min_features)
plot_mean_sd(X_train, score_collector, file_name+"_variance.png", removal_step=removal_step, min_features=min_features)

(mse_train, mse_test, Y_train_predictions, Y_test_predictions) = final_fit_predict(X_train,Y_train,X_test, Y_test, pipeline)
write_stats_file(mse_train, mse_test, pipeline, best_param_feature_reduction_scores,file_name+".txt", train_joined, execution_time=(end_time-start_time) ) #TODO: izvuci skb element zbog _support
write_predictions(file_name + "_train.tsv", stat_labels_train, Y_train_predictions)
write_predictions(file_name + "_test.tsv", stat_labels_test, Y_test_predictions)


