import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
import time
from sklearn.model_selection import train_test_split
from matplotlib import cm

from sklearn import preprocessing
from sklearn import utils
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

def shoot_distance(data):
    plt.figure(figsize=(5,5))
    plt.scatter(data.dist, data.shot_distance, color='blue')
    plt.title("dist and shot_distance")

def drop_data(data,drops):
    for drop in drops:
        data = data.drop(drop,1)
    return data

def make_dummies(data, categorical_vars):
    for var in categorical_vars:
        data = pd.concat([data, pd.get_dummies(data[var], prefix=var)], 1)
        data = data.drop(var,1)
    return data


def lofloss(act,pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act) * sp.log(sp.substract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


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

droped_columns = ['shot_id','team_id','team_name','shot_zone_area','shot_zone_range','shot_zone_basic',\
                  'matchup','lon','lat','seconds_remaining','minutes_remaining', 'shot_distance',\
                  'loc_x','loc_y','game_event_id','game_id','game_date']




X = data.drop(droped_columns, axis=1)
Y = data['shot_made_flag']

X_test, X_train, Y_test, Y_train = train_test_split(X,Y, test_size = 0.2)

raw = drop_data(raw, droped_columns)

categorical_vars = ['action_type','combined_shot_type','shot_type','opponent','period','season']

raw = make_dummies(raw, categorical_vars)



df = raw[pd.notnull(raw['shot_made_flag'])]
submission = raw[pd.isnull(raw['shot_made_flag'])]
submission = submission.drop('shot_made_flag',1)

train = df.drop('shot_made_flag',1)
train_y = df['shot_made_flag']

#lab_enc = preprocessing.LabelEncoder()
#encoded = lab_enc.fit_transform(train)
#train = encoded
#lab_enc = preprocessing.LabelEncoder(train_y)
#encoded = lab_enc.fit_transform
#train_y = encoded


print ('Finding best n_estimators for RandomForestClassifier...')
min_score = 100000
best_n = 0
score_n = []
range_n = np.logspace(0,2,num=3).astype(int)
for n in range_n:
    print("the number of trees : {0}".format(n))
    t1 = time.time()

    rfc_score = 0.

    rfc = RandomForestClassifier(n_estimators=n)
    for train_k, test_k in KFold(len(train), n_folds=10,shuffle=True):
        rfc.fit(train.iloc[train_k], train_y.iloc[train_k])
        pred = rfc.predict(train.iloc[tes_k])
        rfc_score += logloss(train_y.iloc[test_k], pred) /10
        scores_n.append(rfc_score)
        if rfc_score < min_score:
            min_score = rfc_score
            best_n = n

    t2 = time.time()
    print('Done processing {0} trees ({1:3f}sec)'.format(n,t2-t1))
print(best_n, min_score)
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

#shoot_distance(raw)

#scatter_plot_by_category('shot_zone_area')
#scatter_plot_by_category('shot_zone_basic')
#scatter_plot_by_category('shot_zone_range')
