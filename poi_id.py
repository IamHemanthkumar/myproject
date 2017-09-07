#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list=['poi','loan_advances','shared_receipt_with_poi',
              'from_poi_to_this_person','to_messages','from_messages','total_payments',
              'director_fees','restricted_stock_deferred','salary','bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

#####Outlier removal#################
data_dict.pop("TOTAL",0)
###########Dataframe creation##########
import pandas as pd
df = pd.DataFrame(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))
df.set_index(employees, inplace=True)
df_new = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).copy()
df_new1=df_new.fillna(0)

############# new feature creation ####################
proportion1=((df_new1['from_this_person_to_poi']/df_new1['from_messages'])*1.0).fillna(0)
proportion2=((df_new1['from_poi_to_this_person']/df_new1['to_messages'])*1.0).fillna(0)
salary_to_bonus=((df_new1['salary']/df_new1['bonus'])*1.0).fillna(0)
df_new1["salary_to_bonus"]=salary_to_bonus
import numpy as np
df_new1=df_new1.replace([np.inf, -np.inf], 0)
df_new1["proportion1"]=proportion1
df_new1["proportion2"]=proportion2 
print df_new1["salary"].describe()
print df_new1["bonus"].describe()
print df_new1.describe() 
df_dict = df_new1.to_dict('index')
my_dataset = df_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

clft = tree.DecisionTreeClassifier()
nb=GaussianNB()
cslf=SVC(random_state=5)
kbest=SelectKBest(k="all")
scale=MinMaxScaler()
scale.fit(features,labels)
f=scale.transform(features)
kbest.fit(features,labels)
################################### calculating score for features using selectkbest####################################
# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in kbest.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  kbest.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'kbest.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in kbest.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

print features_selected_tuple
######################## using stratifiedshufflesplit with n_splits equal to 1000 which gives highest precision and recall value###############

ss=StratifiedShuffleSplit(n_splits=10, test_size=20, random_state=0)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.metrics import precision_score

#################### settingup parameters for SVC ############################
#################### SVC gives me the best precision and recall value, So my final selection of algorithm is SVC  ####################
parameters = {'cslf__kernel':['rbf'],
         'cslf__C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'cslf__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1,1,10]
             }

###########################scaling the feature and using SVC in the pipeline################################
pipeline=Pipeline(steps=[("scale",scale),("cslf",cslf)])
gs = GridSearchCV(pipeline, parameters,scoring="precision",cv=ss)
gs.fit(features, labels)
print gs.best_estimator_
clf =  gs.best_estimator_

######################## below two algorithms are for comparison purpose, remove the hash if you want to compare ###############
#parameters={'clft__splitter':['best','random'],
#           'clft__criterion':['gini','entropy'],
#           'clft__min_samples_split':[2,5,10,30,50,80,100],
#           'clft__max_features':['auto','sqrt','log2'],
#           'clft__presort':['True','False']}
#pipeline=Pipeline(steps=[("scale",scale),("clft",clft)])
#gt = GridSearchCV(pipeline, parameters,scoring="precision",cv=ss)
#gt.fit(features, labels)
#print gt.best_estimator_
#cl =  gt.best_estimator_


#parameters={}
#pipeline=Pipeline(steps=[("scale",scale),("nb",nb)])
#gb = GridSearchCV(nb, parameters,scoring="precision",cv=ss)
#gb.fit(features, labels)
#print gb.best_estimator_
#gb_clf =  gb.best_estimator_

from tester import test_classifier
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)