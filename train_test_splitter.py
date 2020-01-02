import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def exclude_2015(dataframe):
    test_set = dataframe.loc[dataframe['year'] == 2015]
    whole_cleaned_set = dataframe.loc[dataframe['year'] != 2015]
    return (whole_cleaned_set, test_set)


def bin_age(dataframe):
    dataframe['age_bin'] = pd.cut(dataframe["age"],
                         bins=[0, 45, 89, 133],
                         labels=[1,2,3])
    return dataframe


def assign_categories(dataframe):

    category_list = np.zeros(len(dataframe.index))
    dataframe["temp_category"] = category_list

    for i, row in dataframe.iterrows():
        if(dataframe.at[i,"cultivar"] == "M206" and dataframe.at[i,"year"] == 2014):
            dataframe.at[i,"temp_category"] = 1

        if (dataframe.at[i, "cultivar"] == "M206" and dataframe.at[i, "year"] == 2016):
            dataframe.at[i, "temp_category"] = 2

        if (dataframe.at[i, "cultivar"] == "Nipponbare"):
            dataframe.at[i, "temp_category"] = 3

        if (dataframe.at[i, "cultivar"] == "M401"):
            dataframe.at[i, "temp_category"] = 4

        if (dataframe.at[i, "cultivar"] == "Kitaake"):
            dataframe.at[i, "temp_category"] = 5

        if (dataframe.at[i, "cultivar"] == "CLXL745"):
            dataframe.at[i, "temp_category"] = 6

        if (dataframe.at[i, "cultivar"] == "Sabine"):
            dataframe.at[i, "temp_category"] = 7

    return dataframe


def assign_final(dataframe, compartment):

    category_list = np.zeros(len(dataframe.index))
    dataframe["cv_category"] = category_list
    age_bins = [1,2,3]
    temp_cv = [1,2,3,4,5,6,7]


    combination_dict = {}
    cnt = 1
    for combination in itertools.product(age_bins, temp_cv):
        if compartment == "endosphere" and cnt == 3:
            combination_dict[combination] = 10 # concat with the chronologically next group
        else:
            combination_dict[combination] = cnt
        cnt = cnt +1



    for i, row in dataframe.iterrows():
        combination = [dataframe.at[i, "age_bin"], int(dataframe.at[i, "temp_category"])]
        combination = tuple(combination)
        dataframe.at[i, "cv_category"] = int(combination_dict[combination])

    del dataframe["temp_category"]
    del dataframe["age_bin"]


    return dataframe


def exclude_2014_2016_cali_M206(dataframe):
    train_set = dataframe.loc[dataframe['cultivar'] == "M206"]
    whole_cleaned_set = dataframe.loc[dataframe['cultivar'] != "M206"]
    return (whole_cleaned_set, train_set)


def separate(whole_set, train_set, test_set):
    (train_temp, test_temp) = train_test_split(whole_set, test_size=0.2, stratify=whole_set['cv_category'], shuffle=True)


    train_set = pd.concat([train_set, train_temp])
    test_set = pd.concat([test_set, test_temp], sort=False)



    train_set = shuffle(train_set)
    test_set = shuffle(test_set)
    del test_set["cv_category"]
    return (train_set, test_set)



# * * * * * * * *
# - - - MAIN - - -
# * * * * * * * *
compartment = "rizosphere"
endosphere = "data/endosphere/base.tsv"
rizosphere = "data/rizosphere/base.tsv"
# load data into pandas dataframe
endosphere = pd.read_csv(endosphere,delimiter='\t',encoding='utf-8')
rizosphere = pd.read_csv(rizosphere,delimiter='\t',encoding='utf-8')

if(compartment == "rizosphere"):
    whole_set = rizosphere
else:
    whole_set = endosphere

(whole_set,test_set) = exclude_2015(whole_set)
whole_set = bin_age(whole_set)
whole_set = assign_categories(whole_set)
whole_set = assign_final(whole_set, compartment)
(whole_set,train_set) = exclude_2014_2016_cali_M206(whole_set)
(train_set, test_set) = separate(whole_set, train_set, test_set)
output_location = "data/" + compartment + "/"
train_set.to_csv(output_location+"train.tsv", sep='\t', index=False)
test_set.to_csv(output_location+"test.tsv", sep='\t', index=False)
print("Train: " + str(train_set.shape))
print("Test: " + str(test_set.shape))
