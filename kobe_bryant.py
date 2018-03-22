# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 12:36:00 2018
# =============================================================================
# #My name is Mich Mich !!!
# =============================================================================
@author: Michel
"""


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals.six import StringIO
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
################################Functions######################################

def number_of_occurences(data, column, value):
    return data[column].value_counts()[value]

def number_of_occurences_to_dict(data, column):
    index = []
    values_occurences = []
    for i in data[column].unique():
        index.append(i)
    for i in index:
        values_occurences.append(number_of_occurences(data, column, i))
    return dict(zip(index,values_occurences))
        
def bar_plot(dictionnary):
    fig, ax = plt.subplots()
    ind = np.arange(len(dictionnary))
    y_list = list(dictionnary.values())
    x_list = list(dictionnary.keys())
    x_list.insert(0,'blank')
    ax.bar(ind, y_list, color='purple')
    ax.set_ylabel('Count')
    ax.set_xticklabels(x_list)
    plt.show()
 
def clean_data(data):
    data['shot_made_flag'].fillna(0.5, inplace=True)
    return data

def scatter_plot(data, columns1, columns2, color):
    alpha = 0.02
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.scatter(data[columns1], data[columns2], color=color, alpha=alpha)
    plt.title(str(columns1)+ " " + str(columns2))

def scatter_plot_by_category(feat):
    alpha=0.1
    gs = raw.groupby(feat)
    cs = cm.rainbow(np.linspace(0,1,len(gs)))
    for g,c in zip(gs,cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)
        
        

####################Coding###############

raw = pd.read_csv('data_kb.csv')

data = clean_data(raw)

nona = data[pd.notnull(data['shot_made_flag'])]


#####################Data Shapping#######
    
    
##### Loc_x and Loc_y shapping
raw['dist'] = np.sqrt(raw['loc_x']**2 + raw['loc_y']**2)
loc_x_zero = raw['loc_x'] == 0
raw['angle'] = np.array([0]*len(raw))
raw['angle'][~loc_x_zero] = np.arctan(raw['loc_y'][~loc_x_zero] / raw['loc_x'][~loc_x_zero])
raw['angle'][loc_x_zero] = np.pi /2
#print(raw['angle'])

raw['reamaining_time'] = raw['minutes_remaining'] * 60 + raw['seconds_remaining']

raw['season'] = raw['season'].apply(lambda x : int(x.split('-')[1]))



###################Regression#############

droped_columns = ['shot_made_flag','game_event_id','opponent','matchup','game_date','team_name'\
                  ,'shot_zone_range','shot_zone_basic', 'shot_zone_area','shot_type', 'season',\
                  'combined_shot_type','action_type']

X = data.drop(droped_columns, axis=1)
Y = data['shot_made_flag']

X_test, X_train, Y_test, Y_train = train_test_split(X,Y, test_size = 0.2)


    

######################Ramdon Forest ####################


####################Graphics############################
    
#shot_type = number_of_occurences_to_dict(data, 'combined_shot_type')    
#bar_plot(shot_type)
#print ("shot type " + str(shot_type))

#shot_zone = number_of_occurences_to_dict(data,'shot_zone_area')
#bar_plot(shot_zone)
#print ("shot zone " + str(shot_zone))

#shot_zone_basic = number_of_occurences_to_dict(data, 'shot_zone_basic')
#bar_plot(shot_zone_basic)
#print ("shot_zone_basic " + str(shot_zone_basic))


#shot_made_flag = number_of_occurences_to_dict(data, 'shot_made_flag')
#bar_plot(shot_made_flag)
#print ("shot_made_flag " + str(shot_made_flag))

#scatter_plot(nona, 'loc_x', 'loc_y', 'blue')
#scatter_plot(nona, 'lon', 'lat', 'green')


scatter_plot_by_category('shot_zone_area')
scatter_plot_by_category('shot_zone_basic')
scatter_plot_by_category('shot_zone_range')