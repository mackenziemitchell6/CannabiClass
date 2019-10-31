#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:27:52 2019

@author: mackenziemitchell
"""

#Importing Libraries and Getting df
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve
from sklearn.metrics import auc, classification_report, confusion_matrix
from sklearn import tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.externals.six import StringIO 
import xgboost as xgb
from CannabiFunctions import find_best_k, print_metrics, roc, plot_feature_importances, plot_corr_matrix
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA  
#from Functionss import plot_corr_matrix, print_metrics, find_best_k, roc, plot_feature_importances
from sklearn.linear_model import LogisticRegression
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
with open('Pickles/df.pickle','rb') as file:
    fulldf=pickle.load(file)
import warnings
warnings.filterwarnings('ignore')

#Looking for class imbalance
indicadf=fulldf[fulldf['type']==0]
sativadf=fulldf[fulldf['type']==1]
hybriddf=fulldf[fulldf['type']==2]

sns.distplot(fulldf['type'])
prind=len(indicadf)/len(fulldf)
prsat=len(sativadf)/len(fulldf)
prhyb=len(hybriddf)/len(fulldf)
print('Probability of Indica: {}'.format(prind))
print('Probability of Sativa: {}'.format(prsat))
print('Probability of Hybrid: {}'.format(prhyb))

#Train Test Split
features=fulldf.drop(columns=['name','type'])
trainn=fulldf.drop(columns='name')
selectedfeatures=['thc','Relaxed','Hungry','Sleepy','Depression','Insomnia','Pain','Euphoric','Creative','Energetic','Dry Mouth','Nausea','Uplifted','Fatigue','Focused']
# target=label_binarize(fulldf.type,classes=[0,1,2])
target=fulldf.type
# n_classes=target.shape[1]
                    
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25,random_state=42)
trainx=X_train[selectedfeatures]
testx=X_test[selectedfeatures]

#PCA
pca = PCA()
features=fulldf.drop(columns='name')
transformed = pca.fit_transform(features)
# plt.scatter(transformed[:,0], transformed[:,1],hue='ty');
pca.components_
pcadict={'feat1':transformed[:,0],'feat2':transformed[:,1]}
pcadf=pd.DataFrame(pcadict)
pcadf['target']=fulldf['type']
pcadf.head()
plt.figure(figsize=(10,5))
sns.scatterplot(x='feat1',y='feat2',hue='target',data=pcadf)
plt.savefig('Visualizations/PCAPlot.png', bbox_inches='tight')
pca.mean_
featurespca=pcadf.drop(columns='target')
targetpca=pcadf.target
Xptrain, Xptest, yptrain, yptest = train_test_split(featurespca, targetpca, test_size=0.25,random_state=42)

#Scale Training Features
scaler = StandardScaler()

scaled_data_train = scaler.fit_transform(X_train)
scaled_data_test = scaler.transform(X_test)

#Dummy Model (accuracy: 0.5436)
dummy = DummyClassifier(strategy='most_frequent', random_state=1)

dummy.fit(scaled_data_train, y_train)
basepreds=dummy.predict(scaled_data_test)
sc=dummy.score(scaled_data_test, y_test) 
print("Baseline Model Metrics:")
print_metrics(y_test,basepreds)
plt.figure(figsize=(10,6))
plot_corr_matrix(y_test,basepreds,['Indica','Sativa','Hybrid'],'Baseline')
plt.savefig('Visualizations/BaselineConfuseMatrix.png', bbox_inches='tight')
print(classification_report(y_test,basepreds,target_names=['indica','satica','hybrid']))

#Logistic Regression Baseline(Accuracy: 0.6628)
logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
model_log = logreg.fit(scaled_data_train, y_train)
y_pred=model_log.predict(scaled_data_test)
print_metrics(y_test,y_pred)
plt.figure(figsize=(10,6))
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/LogisticConfuseMatrix.png', bbox_inches='tight')
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))

#Logistic Regression With Selected FEatures (accuracy: 0.6860)
logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
model_log = logreg.fit(df_train, y_train)
y_pred=model_log.predict(df_test)
print_metrics(yptest,y_pred)
plt.figure(figsize=(10,6))
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/LogisticSelectedConfuseMatrix.png', bbox_inches='tight')
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))

#Logistic Regression with PCA (Accuracy: 0.5436)
logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
model_log = logreg.fit(Xptrain, yptrain)
y_pred=model_log.predict(Xptest)
print_metrics(yptest,y_pred)
plt.figure(figsize=(10,6))
plot_corr_matrix(yptest,y_pred,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/LogisticPCAConfuseMatrix.png', bbox_inches='tight')
print(classification_report(yptest,y_pred,target_names=['indica','satica','hybrid']))

#KNN Baseline (Accuracy: 0.5291)
clf1 = KNeighborsClassifier()
clf1.fit(scaled_data_train, y_train)
test_preds = clf1.predict(scaled_data_test)
print_metrics(y_test, test_preds)
print(classification_report(y_test,test_preds,target_names=['indica','satica','hybrid']))
plot_corr_matrix(y_test,test_preds,['Indica','Sativa','Hybrid'])

#KNN With Selected Features (Accuracy: 0.6192)
clf1 = KNeighborsClassifier()
clf1.fit(df_train, y_train)
test_preds = clf1.predict(df_test)
print_metrics(y_test, test_preds)
print(classification_report(y_test,test_preds,target_names=['indica','satica','hybrid']))
confusion_matrix(test_preds,y_test)
plot_corr_matrix(y_test,test_preds,['Indica','Sativa','Hybrid'])

#KNN With PCA (Accuracy: 0.4947)
clf1 = KNeighborsClassifier()
clf1.fit(Xptrain, yptrain)
test_preds = clf1.predict(Xptest)
print_metrics(yptest, test_preds)
print(classification_report(yptest,test_preds,target_names=['indica','satica','hybrid']))
confusion_matrix(test_preds,yptest)
plot_corr_matrix(y_test,test_preds,['Indica','Sativa','Hybrid'])

#Getting best KNN model (k=23, accuracy: 0.6512)
find_best_k(scaled_data_train, y_train, scaled_data_test, y_test)
clf1 = KNeighborsClassifier(n_neighbors=23)
clf1.fit(scaled_data_train, y_train)
test_preds = clf1.predict(scaled_data_test)
print_metrics(y_test, test_preds)
print(classification_report(y_test,test_preds,target_names=['indica','satica','hybrid']))
plt.figure(figsize=(10,5))
plot_corr_matrix(y_test,test_preds,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/BestKKNNConfuseMatrix.png', bbox_inches='tight')

#Finding best k with selected features (k=15, accuracy: .6599)
find_best_k(df_train, y_train, df_test, y_test)
clf1 = KNeighborsClassifier(n_neighbors=17)
clf1.fit(df_train, y_train)
test_preds = clf1.predict(df_test)
print_metrics(y_test, test_preds)
dt_cv_score=cross_val_score(clf1, df_train, y_train)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))
print(classification_report(y_test,test_preds,target_names=['indica','satica','hybrid']))
plot_corr_matrix(y_test,test_preds,['Indica','Sativa','Hybrid'])

#Finding best k with PCA (k=1 accuracy: 0.5233)
find_best_k(Xptrain, yptrain, Xptest, yptest)
clf1 = KNeighborsClassifier(n_neighbors=1)
clf1.fit(Xptrain, yptrain)
test_preds = clf1.predict(Xptest)
print_metrics(yptest, test_preds)
dt_cv_score=cross_val_score(clf1, Xptrain, yptrain)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))
print(classification_report(yptest,test_preds,target_names=['indica','satica','hybrid']))
plot_corr_matrix(yptest,test_preds,['Indica','Sativa','Hybrid'])

#Grid Search for KNN (k=30, weights=uniform, accuracy: 0.6279)
model=KNeighborsClassifier()
dt_cv_score=cross_val_score(model, scaled_data_train, y_train)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))

dt_param_grid = {
    'n_neighbors':list(range(1,31)),
    'weights':['uniform','distance']
}

dt_grid_search = GridSearchCV(model, dt_param_grid, cv=10, return_train_score=True)
dt_grid_search.fit(scaled_data_train, y_train)

dt_gs_training_score = np.mean(dt_grid_search.cv_results_['mean_train_score'])
dt_gs_testing_score = dt_grid_search.score(scaled_data_test, y_test)

print("Mean Training Score: {:.4}%".format(dt_gs_training_score * 100))
print("Mean Testing Score: {:.4}%".format(dt_gs_testing_score * 100))
print("Best Parameter Combination Found During Grid Search:")
dt_grid_search.best_params_
clf1 = KNeighborsClassifier(n_neighbors=30, weights='uniform')
clf1.fit(scaled_data_train, y_train)
test_preds = clf1.predict(scaled_data_test)
print_metrics(y_test, test_preds)
dt_cv_score=cross_val_score(clf1, scaled_data_train, y_train)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))
print(classification_report(y_test,test_preds,target_names=['indica','satica','hybrid']))
plt.figure(figsize=(10,5))
plot_corr_matrix(y_test,test_preds,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/GridKNNConfuseMatrix.png', bbox_inches='tight')

#Grid Search for KNN with selected features (k=28, uniform, accuracy: 0.6628)
dt_cv_score=cross_val_score(model, df_train, y_train)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))

dt_param_grid = {
    'n_neighbors':list(range(1,31)),
    'weights':['uniform','distance']
}

dt_grid_search = GridSearchCV(model, dt_param_grid, cv=10, return_train_score=True)
dt_grid_search.fit(df_train, y_train)

dt_gs_training_score = np.mean(dt_grid_search.cv_results_['mean_train_score'])
dt_gs_testing_score = dt_grid_search.score(df_test, y_test)

print("Mean Training Score: {:.4}%".format(dt_gs_training_score * 100))
print("Mean Testing Score: {:.4}%".format(dt_gs_testing_score * 100))
print("Best Parameter Combination Found During Grid Search:")
dt_grid_search.best_params_
clf1 = KNeighborsClassifier(n_neighbors=28, weights='uniform')
clf1.fit(df_train, y_train)
test_preds = clf1.predict(df_test)
print_metrics(y_test, test_preds)
dt_cv_score=cross_val_score(clf1, df_train, y_train)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))
print(classification_report(y_test,test_preds,target_names=['indica','satica','hybrid']))
plot_corr_matrix(y_test,test_preds,['Indica','Sativa','Hybrid'])

#Grid search with PCA(k=28,distance, accuracy: 0.5378)
dt_cv_score=cross_val_score(model, Xptrain, yptrain)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))

dt_param_grid = {
    'n_neighbors':list(range(1,31)),
    'weights':['uniform','distance']
}

dt_grid_search = GridSearchCV(model, dt_param_grid, cv=10, return_train_score=True)
dt_grid_search.fit(Xptrain, yptrain)

dt_gs_training_score = np.mean(dt_grid_search.cv_results_['mean_train_score'])
dt_gs_testing_score = dt_grid_search.score(Xptest, yptest)

print("Mean Training Score: {:.4}%".format(dt_gs_training_score * 100))
print("Mean Testing Score: {:.4}%".format(dt_gs_testing_score * 100))
print("Best Parameter Combination Found During Grid Search:")
dt_grid_search.best_params_
clf1 = KNeighborsClassifier(n_neighbors=10,weights='distance')
clf1.fit(Xptrain, yptrain)
test_preds = clf1.predict(Xptest)
print_metrics(yptest, test_preds)
dt_cv_score=cross_val_score(clf1, Xptrain, yptrain)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))
print(classification_report(yptest,test_preds,target_names=['indica','satica','hybrid']))
plot_corr_matrix(yptest,test_preds,['Indica','Sativa','Hybrid'])


#Decision Tree Baseline (accuracy: 0.5436)
classifier1 = DecisionTreeClassifier()  
classifier1.fit(scaled_data_train, y_train) 
y_pred = classifier1.predict(scaled_data_test)
y_score = classifier1.score(scaled_data_test, y_test)
print('Accuracy: ', y_score)

# Compute the average precision score
micro_precision = precision_score(y_pred, y_test, average='micro')
print('Micro-averaged precision score: {0:0.2f}'.format(
      micro_precision))

macro_precision = precision_score(y_pred, y_test, average='macro')
print('Macro-averaged precision score: {0:0.2f}'.format(
      macro_precision))

per_class_precision = precision_score(y_pred, y_test, average=None)
print('Per-class precision score:', per_class_precision)
print_metrics(y_test,y_pred)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
#roc(classifier1,scaled_data_train,scaled_data_test,y_train,y_test,n_classes)
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])

#Decision Tree with selected features (accuracy:0.5436)
classifier2 = DecisionTreeClassifier()  
classifier2.fit(df_train, y_train) 
y_pred = classifier2.predict(df_test)
y_score = classifier2.score(df_test, y_test)
print('Accuracy: ', y_score)

# Compute the average precision score
micro_precision = precision_score(y_pred, y_test, average='micro')
print('Micro-averaged precision score: {0:0.2f}'.format(
      micro_precision))

macro_precision = precision_score(y_pred, y_test, average='macro')
print('Macro-averaged precision score: {0:0.2f}'.format(
      macro_precision))

per_class_precision = precision_score(y_pred, y_test, average=None)
print('Per-class precision score:', per_class_precision)
print_metrics(y_test,y_pred)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
# roc(classifier1,df_train,df_test,y_train,y_test,n_classes)
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])

#Decision Tree with PCA (accuracy: 0.4651)
classifier1.fit(Xptrain, yptrain) 
y_pred = classifier1.predict(Xptest)
y_score = classifier1.score(Xptest, yptest)
print('Accuracy: ', y_score)

# Compute the average precision score
micro_precision = precision_score(y_pred, yptest, average='micro')
print('Micro-averaged precision score: {0:0.2f}'.format(
      micro_precision))

macro_precision = precision_score(y_pred, yptest, average='macro')
print('Macro-averaged precision score: {0:0.2f}'.format(
      macro_precision))

per_class_precision = precision_score(y_pred, yptest, average=None)
print('Per-class precision score:', per_class_precision)
print_metrics(yptest,y_pred)
print(classification_report(yptest,y_pred,target_names=['indica','satica','hybrid']))
# roc(classifier1,df_train,df_test,y_train,y_test,n_classes)
plot_corr_matrix(yptest,y_pred,['Indica','Sativa','Hybrid'])

#Grid search Decision Tree: All features (criterion='entropy',max_depth=3,min_samples_leaf=1,min_samples_split=2, accuracy: 0.6483)
model=DecisionTreeClassifier()
dt_cv_score=cross_val_score(model, scaled_data_train, y_train)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))

dt_param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6]   
}

dt_grid_search = GridSearchCV(model, dt_param_grid, cv=10, return_train_score=True)
dt_grid_search.fit(scaled_data_train, y_train)

dt_gs_training_score = np.mean(dt_grid_search.cv_results_['mean_train_score'])
dt_gs_testing_score = dt_grid_search.score(scaled_data_test, y_test)

print("Mean Training Score: {:.4}%".format(dt_gs_training_score * 100))
print("Mean Testing Score: {:.4}%".format(dt_gs_testing_score * 100))
print("Best Parameter Combination Found During Grid Search:")
dt_grid_search.best_params_

classifier3 = DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=1,min_samples_split=2)  
classifier3.fit(scaled_data_train, y_train) 

y_pred = classifier3.predict(scaled_data_test)

acc = accuracy_score(y_test,y_pred)
print("Accuracy is: {}".format(acc))

micro_precision = precision_score(y_pred, y_test, average='micro')
print('Micro-averaged precision score: {0:0.2f}'.format(
      micro_precision))

macro_precision = precision_score(y_pred, y_test, average='macro')
print('Macro-averaged precision score: {0:0.2f}'.format(
      macro_precision))

per_class_precision = precision_score(y_pred, y_test, average=None)
print('Per-class precision score:', per_class_precision)
print_metrics(y_test,y_pred)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
# roc(classifier3,scaled_data_train,scaled_data_test,y_train,y_test,n_classes)
plt.figure(figsize=(10,5))
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/DecisionTreeGridSearch.png')

#Grid search decision tree with selected features (criterion='entropy',max_depth=3,min_samples_leaf=1,min_samples_split=2, accuracy: 0.6483)
model=DecisionTreeClassifier()
dt_cv_score=cross_val_score(model, df_train, y_train)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))

dt_param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6]   
}

dt_grid_search = GridSearchCV(model, dt_param_grid, cv=10, return_train_score=True)
dt_grid_search.fit(df_train, y_train)

dt_gs_training_score = np.mean(dt_grid_search.cv_results_['mean_train_score'])
dt_gs_testing_score = dt_grid_search.score(df_test, y_test)

print("Mean Training Score: {:.4}%".format(dt_gs_training_score * 100))
print("Mean Testing Score: {:.4}%".format(dt_gs_testing_score * 100))
print("Best Parameter Combination Found During Grid Search:")
dt_grid_search.best_params_
classifier3 = DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=1,min_samples_split=2)  
classifier3.fit(df_train, y_train) 
y_pred = classifier3.predict(df_test)
acc = accuracy_score(y_test,y_pred)
print("Accuracy is: {}".format(acc))

micro_precision = precision_score(y_pred, y_test, average='micro')
print('Micro-averaged precision score: {0:0.2f}'.format(
      micro_precision))

macro_precision = precision_score(y_pred, y_test, average='macro')
print('Macro-averaged precision score: {0:0.2f}'.format(
      macro_precision))

per_class_precision = precision_score(y_pred, y_test, average=None)
print('Per-class precision score:', per_class_precision)
print_metrics(y_test,y_pred)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
# roc(classifier3,df_train,df_test,y_train,y_test,n_classes)
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])

#Grid search decision tree with PCA (criterion='entropy',max_depth=3,min_samples_leaf=1,min_samples_split=2, accuracy: 0.5581)
model=DecisionTreeClassifier()
dt_cv_score=cross_val_score(model, Xptrain, yptrain)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))

dt_param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6]   
}

dt_grid_search = GridSearchCV(model, dt_param_grid, cv=10, return_train_score=True)
dt_grid_search.fit(Xptrain, yptrain)

dt_gs_training_score = np.mean(dt_grid_search.cv_results_['mean_train_score'])
dt_gs_testing_score = dt_grid_search.score(Xptest, yptest)

print("Mean Training Score: {:.4}%".format(dt_gs_training_score * 100))
print("Mean Testing Score: {:.4}%".format(dt_gs_testing_score * 100))
print("Best Parameter Combination Found During Grid Search:")
dt_grid_search.best_params_
classifier3 = DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=1,min_samples_split=2)  
classifier3.fit(Xptrain, yptrain) 
y_pred = classifier3.predict(Xptest)
acc = accuracy_score(yptest,y_pred)
print("Accuracy is: {}".format(acc))

micro_precision = precision_score(y_pred, yptest, average='micro')
print('Micro-averaged precision score: {0:0.2f}'.format(
      micro_precision))

macro_precision = precision_score(y_pred, yptest, average='macro')
print('Macro-averaged precision score: {0:0.2f}'.format(
      macro_precision))

per_class_precision = precision_score(y_pred, yptest, average=None)
print('Per-class precision score:', per_class_precision)
print_metrics(yptest,y_pred)
print(classification_report(yptest,y_pred,target_names=['indica','satica','hybrid']))
# roc(classifier3,df_train,df_test,y_train,y_test,n_classes)
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])

#Random Forest Baseline (accuracy: 0.5494)
tree_clf = DecisionTreeClassifier() 
tree_clf.fit(scaled_data_train, y_train)
y_pred=tree_clf.predict(scaled_data_test)
# roc(tree_clf,scaled_data_train,scaled_data_test,y_train,y_test,n_classes)
plot_feature_importances(tree_clf, pd.DataFrame(X_train), X_test, y_train, y_test)
print_metrics(y_pred,y_test)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])

#Random Forest w Selected features (accuracy: 0.6192)
tree_clf = DecisionTreeClassifier(criterion = "gini", max_depth = 5) 
tree_clf.fit(df_train, y_train)
y_preds=tree_clf.predict(df_test)
# roc(tree_clf,df_train,df_test,y_train,y_test,n_classes)
plot_feature_importances(tree_clf, df_train,  X_test, y_train, y_test)
print_metrics(y_preds,y_test)
print(classification_report(y_test,y_preds,target_names=['indica','satica','hybrid']))
plot_corr_matrix(y_test,y_preds,['Indica','Sativa','Hybrid'])

#Random forest with PCA (accuracy: 0.5262)
tree_clf = DecisionTreeClassifier(criterion = "gini", max_depth = 5) 
tree_clf.fit(Xptrain, yptrain)
y_preds=tree_clf.predict(Xptest)
# roc(tree_clf,df_train,df_test,y_train,y_test,n_classes)
plot_feature_importances(tree_clf, Xptrain, Xptest, yptrain, yptest)
print_metrics(y_preds,yptest)
print(classification_report(yptest,y_preds,target_names=['indica','satica','hybrid']))
plot_corr_matrix(y_test,y_preds,['Indica','Sativa','Hybrid'])

#forest with different parameters (accuracy: 0.6570)
forest = RandomForestClassifier(n_estimators=100, max_depth= 5)
forest.fit(scaled_data_train, y_train)
y_pred=forest.predict(scaled_data_test)
train_score=forest.score(scaled_data_train,y_train)
test_score=forest.score(scaled_data_test,y_test)
print('Training Score: {}'.format(train_score))
print('Testing Score: {}'.format(test_score))
print_metrics(y_pred,y_test)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
plt.figure(figsize=(10,5))
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/ForrestAllDataMtx.png')
plot_feature_importances(forest,pd.DataFrame(X_train),X_test, y_train, y_test)

#forest with different parameters and selected features (accuracy: 0.6715)
forest = RandomForestClassifier(n_estimators=100, max_depth= 5)
forest.fit(df_train, y_train)
y_pred=forest.predict(df_test)
train_score=forest.score(df_train,y_train)
test_score=forest.score(df_test,y_test)
print('Training Score: {}'.format(train_score))
print('Testing Score: {}'.format(test_score))
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
print_metrics(y_pred,y_test)
plot_feature_importances(forest,pd.DataFrame(df_train), X_test, y_train, y_test)
plt.figure(figsize=(10,6))
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/SelectedForestMtx.png')

#forest with different parameters and PCA (accuracy: 0.5610)
forest = RandomForestClassifier(n_estimators=100, max_depth= 5)
forest.fit(Xptrain, yptrain)
y_pred=forest.predict(Xptest)
train_score=forest.score(Xptrain,yptrain)
test_score=forest.score(Xptest,yptest)
print('Training Score: {}'.format(train_score))
print('Testing Score: {}'.format(test_score))
print(classification_report(yptest,y_pred,target_names=['indica','sativa','hybrid']))
print_metrics(y_pred,yptest)
plot_feature_importances(forest,pd.DataFrame(Xptrain), Xptest, yptrain, yptest)
plot_corr_matrix(yptest,y_pred,['Indica','Sativa','Hybrid'])

#Grid search forest (criterion='gini', max_depth=6,min_samples_leaf=5,min_samples_split=5,n_estimators=8, accuracy: 0.6512)
dt_clf=RandomForestClassifier()
dt_cv_score=cross_val_score(dt_clf, scaled_data_train, y_train)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'n_estimators':[1,2,3,4,5,6,7,8,9],
    'max_depth': [None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}
dt_grid_search = GridSearchCV(dt_clf, dt_param_grid, cv=10, return_train_score=True)
dt_grid_search.fit(scaled_data_train, y_train)
dt_gs_training_score = np.mean(dt_grid_search.cv_results_['mean_train_score'])
dt_gs_testing_score = dt_grid_search.score(scaled_data_test, y_test)

print("Mean Training Score: {:.4}%".format(dt_gs_training_score * 100))
print("Mean Testing Score: {:.4}%".format(dt_gs_testing_score * 100))
print("Best Parameter Combination Found During Grid Search:")
dt_grid_search.best_params_

forest3 = RandomForestClassifier(criterion='gini', max_depth=6,min_samples_leaf=5,min_samples_split=5,n_estimators=8)
forest3.fit(scaled_data_train, y_train)
y_pred=forest3.predict(scaled_data_test)
y_pred=forest3.predict(scaled_data_test)
print_metrics(y_test,y_pred)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
plot_feature_importances(forest3,pd.DataFrame(X_train),X_test, y_train, y_test)
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])
print("Training Score: ",forest3.score(scaled_data_train,y_train))
print("Testing Score: ",forest3.score(scaled_data_test,y_test))

#Grid Search Forest selected features (criterion='gini', max_depth=4,min_samples_leaf=5,min_samples_split=10,n_estimators=5 accuracy: 0.6541)
dt_clf=RandomForestClassifier()
dt_cv_score=cross_val_score(dt_clf, df_train, y_train)
meandtcv=dt_cv_score.mean()
print("Mean Cross Validation Score: {:.4}%".format(meandtcv * 100))
dt_grid_search = GridSearchCV(dt_clf, dt_param_grid, cv=10, return_train_score=True)
dt_grid_search.fit(df_train, y_train)
dt_gs_training_score = np.mean(dt_grid_search.cv_results_['mean_train_score'])
dt_gs_testing_score = dt_grid_search.score(df_test, y_test)

print("Mean Training Score: {:.4}%".format(dt_gs_training_score * 100))
print("Mean Testing Score: {:.4}%".format(dt_gs_testing_score * 100))
print("Best Parameter Combination Found During Grid Search:")
dt_grid_search.best_params_
forest3 = RandomForestClassifier(criterion='gini', max_depth=4,min_samples_leaf=5,min_samples_split=10,n_estimators=5)
forest3.fit(scaled_data_train, y_train)
y_pred=forest3.predict(scaled_data_test)
y_pred=forest3.predict(scaled_data_test)
print_metrics(y_test,y_pred)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
plot_feature_importances(forest3,pd.DataFrame(X_train), X_test, y_train, y_test)
plt.figure(figsize=(10,5))
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/SelectedForestGridSearch.png')
print("Training Score: ",forest3.score(scaled_data_train,y_train))
print("Testing Score: ",forest3.score(scaled_data_test,y_test))


#SVM Baseline (accuracy: 0.6715)
clf = svm.SVC(kernel='linear')
clf.fit(scaled_data_train, y_train)
y_pred=clf.predict(scaled_data_test)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
plt.figure(figsize=(10,6))
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/SVMmtx.png')
print_metrics(y_test,y_pred)

#SVM PCA (accuracy: 0.5436)
clf = svm.SVC(kernel='linear')
clf.fit(Xptrain, yptrain)
y_pred=clf.predict(Xptest)
print(classification_report(yptest,y_pred,target_names=['indica','satica','hybrid']))
plt.figure(figsize=(10,6))
plot_corr_matrix(yptest,y_pred,['Indica','Sativa','Hybrid'])
plt.savefig('Visualizations/SVMPCAmtx.png')
print_metrics(yptest,y_pred)

#Naive Bayes Baseline
from sklearn.naive_bayes import BernoulliNB
clf=BernoulliNB()
clf.fit(scaled_data_train, y_train)
y_pred=clf.predict(scaled_data_test)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
clf=GaussianNB()
clf.fit(scaled_data_train, y_train)
y_pred=clf.predict(scaled_data_test)
print(classification_report(y_test,y_pred,target_names=['indica','satica','hybrid']))
plot_corr_matrix(y_test,y_pred,['Indica','Sativa','Hybrid'])
scaled_data_train=pd.DataFrame(scaled_data_train)
scaled_data_test=pd.DataFrame(scaled_data_test)
print_metrics(y_test,y_pred)







