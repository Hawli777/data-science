#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import pprint
import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("../tools/")

### Task 1: Select what features you'll use.

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)

# Explore dataset

#missing data
data_points = len(data_dict)
poi_count = 0
non_poi_count = 0
missing_value_map = { 'bonus': {'count':0, 'poi':0}, 'deferral_payments': {'count':0, 'poi':0},
    'deferred_income': {'count':0, 'poi':0},'director_fees': {'count':0, 'poi':0}, 
    'exercised_stock_options': {'count':0, 'poi':0}, 'total_payments': {'count':0, 'poi':0},
    'expenses': {'count':0, 'poi':0}, 'loan_advances': {'count':0, 'poi':0},
    'long_term_incentive': {'count':0, 'poi':0}, 'restricted_stock_deferred': {'count':0, 'poi':0},
    'other': {'count':0, 'poi':0}, 'restricted_stock': {'count':0, 'poi':0}, 
    'total_stock_value': {'count':0, 'poi':0}, 'salary': {'count':0, 'poi':0}, 
    'email_address': {'count':0, 'poi':0}, 'from_messages': {'count':0, 'poi':0}, 
    'from_poi_to_this_person': {'count':0, 'poi':0}, 'shared_receipt_with_poi': {'count':0, 'poi':0},
    'from_this_person_to_poi': {'count':0, 'poi':0}, 'to_messages': {'count':0, 'poi':0} }


for person, features in data_dict.iteritems():    
    isPoi = False    
    if features['poi'] == True:
        poi_count += 1
        isPoi = True
    else:
        non_poi_count += 1
    for name, value in features.iteritems():         
        if value == 'NaN':
            missing_value_map[name]['count'] += 1
            if isPoi:
                missing_value_map[name]['poi'] += 1

print "数据点总数:\t", data_points 
print "POI数量:\t\t", poi_count
print "非POI数量:\t", non_poi_count
print "POI率:\t\t", poi_count/data_points
print "特征总数:\t", len(data_dict[data_dict.keys()[0]])


significant_missing_values = []
significant_poi_values = []

#print "{:<25} {:<20} {:<10}".format('Feature','missing','poi')
for feature, values in missing_value_map.iteritems():
    missing_ratio = values['count']/data_points
    if missing_ratio > 0.5:
        significant_missing_values.append(feature)
    poi_ratio = values['poi']/poi_count
    if poi_ratio > 0.5:
        significant_poi_values.append(feature)
    #print "{:<25} {:<20} {:<10}".format(feature, values['count'], values['poi'])

print "缺失值超过50%的特征:", significant_missing_values
print "POI中缺失值超过50%的特征:", significant_poi_values



financial_features = ['salary', 'total_payments', 'bonus', 
                      'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 'restricted_stock']
#'email_address',
email_features = ['to_messages', 'from_poi_to_this_person', 
                  'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list = ['poi'] + financial_features + email_features
df = df[features_list]
print "财务特征数量:\t", len(financial_features)
print "邮件特征数量:\t", len(email_features)

from sklearn.preprocessing import Imputer

# 为财务特征缺失值填充为0 
df[financial_features] = df[financial_features].fillna(0)

# 为邮件特征填充均值
imp = Imputer(missing_values='NaN', strategy = 'mean', axis=0)

df_poi = df[df['poi'] == True];
df_nonpoi = df[df['poi']==False]

df_poi.loc[:, email_features] = imp.fit_transform(df_poi.loc[:,email_features]);
df_nonpoi.loc[:, email_features] = imp.fit_transform(df_nonpoi.loc[:,email_features]);
df = df_poi.append(df_nonpoi)

### Task 2: Remove outliers

# 查找是否有其他异常数据
IQR = df.quantile(q=0.75) - df.quantile(q=0.25)
first_quartile = df.quantile(q=0.25)
third_quartile = df.quantile(q=0.75)
outliers = df[(df>(third_quartile + 1.5*IQR) ) | (df<(first_quartile - 1.5*IQR) )].count(axis=1)
outliers.sort_values(axis=0, ascending=False, inplace=True)
outliers.head(10)

# 去除财务表格上的非poi异常值
df.drop(axis=0, labels=['TOTAL','THE TRAVEL AGENCY IN THE PARK','FREVERT MARK A', 'LAVORATO JOHN J','KEAN STEVEN J', 
                        'WHALLEY LAWRENCE G', 'KITCHEN LOUISE','BAXTER JOHN C'], inplace=True)

# 查看剔除后的数据点
len(df)
df['poi'].value_counts()

### Task 3: Create new feature(s)
# Add the new email features to the dataframe
df['to_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['shared_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']

features_list.append('to_poi_ratio')
features_list.append('from_poi_ratio')
features_list.append('shared_poi_ratio')

# Create the new financial features and add to the dataframe
df['bonus_to_salary'] = df['bonus'] / df['salary']
df['bonus_to_total'] = df['bonus'] / df['total_payments'] 

features_list.append('bonus_to_salary')
features_list.append('bonus_to_total') 

from sklearn.preprocessing import scale
# Fill any NaN financial data with a 0
df.fillna(value= 0, inplace=True)

# Create a copy of the dataframe and normalize it to zero mean and unit variance
scaled_df = df.copy()
scaled_df.iloc[:,1:] = scale(scaled_df.iloc[:,1:])


# Send the dataset from dataframe to dictionary for tester.py
my_dataset = scaled_df.to_dict(orient='index')

# Fill any NaN financial data with a 0
df.fillna(value= 0, inplace=True)

# Create a copy of the dataframe and normalize it to zero mean and unit variance
scaled_df = df.copy()
scaled_df.iloc[:,1:] = scale(scaled_df.iloc[:,1:])

# Send the dataset from dataframe to dictionary for tester.py
my_dataset = scaled_df.to_dict(orient='index')

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
import tester

# Create the classifier, GaussianNB has no parameters to tune
clf = GaussianNB()
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

clf = DecisionTreeClassifier()
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

clf = SVC(kernel='linear')
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

clf = KMeans(n_clusters=2)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from feature_format import featureFormat, targetFeatureSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

data_dict = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data_dict)

from sklearn.model_selection import GridSearchCV

n_features = np.arange(1, len(features_list))


# Create a pipeline with feature selection and classifier
tree_pipe = Pipeline([
    ('select_features', SelectKBest(k=19)),
    ('classify', DecisionTreeClassifier()),
])

# Define the configuration of parameters to test with the 
# Decision Tree Classifier
param_grid = dict(classify__criterion = ['gini', 'entropy'] , 
                  classify__min_samples_split = [2, 4, 6, 8, 10, 20],
                  classify__max_depth = [None, 5, 10, 15, 20],
                  classify__max_features = [None, 'sqrt', 'log2', 'auto'])

# Use GridSearchCV to find the optimal hyperparameters for the classifier
tree_clf = GridSearchCV(tree_pipe, param_grid = param_grid, cv=10)
tree_clf.fit(features, labels)

# Get the best algorithm hyperparameters for the Decision Tree
tree_clf.best_params_

# Create the classifier with the optimal hyperparameters as found by GridSearchCV
tree_clf = Pipeline([
    ('select_features', SelectKBest(k=19)),
    ('classify', DecisionTreeClassifier(criterion='entropy', max_depth=20, max_features=None, min_samples_split=4))
])

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# Test the classifier using tester.py
tester.dump_classifier_and_data(tree_clf, my_dataset, features_list)
tester.main()



