# -*- coding: utf-8 -*-
"""
@author: Xianqiao Li & Liye Pan
"""

import pandas as pd
import numpy as np
import os, time
import itertools
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, recall_score, precision_score


# Set paths
unix_working_dir = '/media/3tb/ClearpoolData/NN with SMOKE+excluding IOC'
windows_working_dir = '/media/3tb/ClearpoolData/Clearpool Project/NN_Loop'
unix_path = '/media/3tb/ClearpoolData/Neural_Networks_Program'

File_name = 'Algorithm_Training_Result.txt'

# Combine Results
os.chdir(unix_working_dir)

File_directory = os.path.join(unix_path, File_name)

df1 = pd.read_csv('LEVL_first hour result.csv')
df2 = pd.read_csv('Basic_Liquidity_LEVL_Clean.csv')

df2.set_index('OrderID', inplace= True)
df1.set_index('OrderID', inplace= True)

df2 = df2.reindex(df1.index)

combined = pd.concat([df1, df2], axis=1)

combined['StartTime'] = pd.to_datetime(combined['StartTime'])

#excluding IOC
combined = combined[combined['TIF']!='IOC']

# Pre-process data
def preprocess_data(combined, Cols):
    
    #transform start time into 10 binary variables
    quantiles = combined['StartTime'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).tolist()

    for i in range(0,10):
        if i == 0:
            combined['StartTime_Q'+str(i+1)] = np.where(combined['StartTime'] <= quantiles[i],1,0)
        elif i < 8:
            combined['StartTime_Q'+str(i+1)] = np.where((combined['StartTime'] > quantiles[i-1]) & (combined['StartTime'] <= quantiles[i]),1,0)
        else:
            combined['StartTime_Q'+str(i+1)] = np.where(combined['StartTime'] > quantiles[i-1],1,0)
            
    #get dummies
    combined_dummies = pd.get_dummies(combined[Cols])

    #split by date
    combined_training = combined_dummies[combined_dummies['TradeDate']<=20170616]
    combined_testing = combined_dummies[combined_dummies['TradeDate']>20170616]

    #replace NaN
    combined_training.fillna(0, inplace = True)
    combined_testing.fillna(0, inplace = True)

    #scaling
    scaling_variables = [
     'LmtPrice',
     'Size',
     'MinExecQty',
     'LotSize',
     'Adv20d',
     'DayVolume',
     'ADFVolume',
     'Previous_Day_Adv20d',
     'Previous_Day_DayVolume',
     'Previous_Day_ADFVolume',
     'One_Week_Avg_DayVolume',
     'One_Week_Avg_ADFVolume']

    for variable in scaling_variables:
        Min = combined_training[variable].min()
        Max = combined_training[variable].max()
        combined_training[variable] = (combined_training[variable]-Min)/(Max-Min)
        combined_testing[variable] = (combined_testing[variable]-Min)/(Max-Min)
        
    return combined_training, combined_testing


# Train and test split
def train_test_split(combined, Cols):
    
    combined_training, combined_testing = preprocess_data(combined, Cols)
    
    cols = combined_training.columns
    X_test = combined_testing[cols[2:]]
    X_test.fillna(0, inplace = True)
    y_test = combined_testing[cols[0]]

    X_train = combined_training[cols[2:]]
    X_train.fillna(0, inplace = True)
    y_train = combined_training[cols[0]]
    
    return X_train, y_train, X_test, y_test


# Hidden_layer_combo Construction
def hidden_layers():
    
    # Full ranges
    t1 = tuple(x for x in xrange(20, 65, 5))
    t2 = tuple(y for y in xrange(10, 70, 10))
    t3 = tuple(z for z in xrange(2, 12, 2))
    
    hidden_layer_sizes = list(itertools.product(t1,t2,t3))
    
    activation_list = ('logistic', 'tanh', 'relu')
    
    hidden_activation_combo = list(itertools.product(hidden_layer_sizes, activation_list))
    
    return hidden_activation_combo


# Fit the Model
def classifier_and_metrics(X_train, y_train, X_test, y_test):
    
    for j in hidden_layers():

    	print j
        
        clf = MLPClassifier(hidden_layer_sizes=j[0], activation=j[1])
    
        clf.fit(X_train, y_train)
                
        clf_predict = clf.predict(X_test)
        
        clf_proba = clf.predict_proba(X_test)[:,1]
    
        clf_fpr, clf_tpr, clf_thresholds = roc_curve(y_test, clf_proba)
        
        clf_auc = auc(clf_fpr, clf_tpr)
        
        clf_recall = recall_score(y_test, clf_predict)
        
        clf_precision = precision_score(y_test, clf_predict)
        
        clf_accuracy = clf.score(X_test, y_test)
        
        print "Accuracy: %f \n" % clf_accuracy
        
        print "AUC: %f \n" % clf_auc
        
        print "Recall Score: %f \n" % clf_recall
        
        print "Precision Score: %f \n" % clf_precision
        
        print "Confusion matrix:"
        conf_mat_df = pd.crosstab(y_test, clf_predict)
        print conf_mat_df
        
        with open(File_directory, 'a') as f:
            f.write('Parameters: {}\n\nAccuracy: {}\n\nAUC: {}\n\nRecall Score: {}\n\nPrecision Score: {}\n\nConfusion Matrix: \n{}\n\n#------------#\n\n'\
                    .format(j, clf_accuracy, clf_auc, clf_recall, clf_precision, conf_mat_df))


# SMOTE
def SMOTE_method(combined, Cols, kn, mn):
    
    X_train, y_train, X_test, y_test = train_test_split(combined, Cols)
    
    global sm
    
    sm = SMOTE(random_state=33, kind='borderline2', k_neighbors=kn, m_neighbors=mn, n_jobs=-1) # -1: use all cores!!!!!!!!!!!!
    
    X_train, y_train = sm.fit_sample(X_train, y_train)
    
    classifier_and_metrics(X_train, y_train, X_test, y_test)


# SMOTE+Tomek
def SMOTE_Tomek_method(sm, combined, Cols):
    
    X_train, y_train, X_test, y_test = train_test_split(combined, Cols)
    
    st = SMOTETomek(random_state=33, smote=sm)
    
    X_train, y_train = st.fit_sample(X_train, y_train)
    
    classifier_and_metrics(X_train, y_train, X_test, y_test)


# SMOTE+ENN
def SMOTE_ENN_method(sm, combined, Cols, nn, ks):
    
    X_train, y_train, X_test, y_test = train_test_split(combined, Cols)
        
    enn = EditedNearestNeighbours(n_neighbors=nn, kind_sel=ks)
    
    st = SMOTEENN(random_state=33, smote=sm, enn=enn)
    
    X_train, y_train = st.fit_sample(X_train, y_train)
    
    classifier_and_metrics(X_train, y_train, X_test, y_test)


# Mix_pool Construction
def Mix_pool():
    k_neighbors = [5, 10, 15]
    m_neighbors = [5, 10, 15]
    n_neighbors = [2, 3, 5]
    kind_sel = ['all', 'mode']
    Mix_pool = list(itertools.product(k_neighbors, m_neighbors, n_neighbors, kind_sel))
    return Mix_pool


# Start the process
if __name__ == '__main__':
    
    M = Mix_pool()
    c = 1
    t0 = time.time()
    
    for m in M:   
        kn, mn, nn, ks =  m[0], m[1], m[2], m[3]
        
        print 'Training with SMOTE method...'
        with open(File_directory, 'a') as g:
            g.write('SMOTE_method\n\n{}\n\n'.format(m))
        SMOTE_method(combined, Cols, kn, mn)
        time.sleep(1)
        
        print 'Training with SMOTE_Tomek method...'
        with open(File_directory, 'a') as g:
            g.write('SMOTE_Tomek_method\n\n{}\n\n'.format(m))
        SMOTE_Tomek_method(sm, combined, Cols)
        time.sleep(1)
    
        print 'Training with SMOTE_ENN method...'
        with open(File_directory, 'a') as g:
            g.write('SMOTE_ENN_method\n\n{}\n\n'.format(m))
        SMOTE_ENN_method(sm, combined, Cols, nn, ks)
        time.sleep(1)
        
        with open(File_directory, 'a') as g:
            g.write('#---------{}/{} cycles---------#\n\n'.format(c, len(Mix_pool())))
        
        print '{}/{} cycles finished; Run time: {} secs'.format(c, len(Mix_pool()), int(time.time()-t0))

        c+=1        
    print 'Run time: {} secs'.format(int(time.time()-t0))