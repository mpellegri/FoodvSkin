# Import common libraries
from numpy.random import rand
from numpy import ones, zeros, concatenate
from pandas import read_csv, DataFrame

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

Food_all = read_csv('./csv_files_K/Food_My18Features.csv')
People_all = read_csv('./csv_files_K/People_My18Features.csv')
Food=Food_all.ix[:,2:]
People=People_all.ix[:,2:]
#Food = read_csv('Food_Features.csv')
#People = read_csv('People_Features.csv')

cTrainF = rand(len(Food)) > .5
cTestF = ~cTrainF

cTrainP = rand(len(People)) > .5
cTestP = ~cTrainP

TrainX = concatenate([People[cTrainP], Food[cTrainF]])
TestX = concatenate([People[cTestP], Food[cTestF]])

TrainY = concatenate([zeros(len(People[cTrainP])), ones(len(Food[cTrainF]))])
TestY = concatenate([zeros(len(People[cTestP])), ones(len(Food[cTestF]))])

tree = DecisionTreeClassifier(max_depth=None, min_samples_split=1, 
                              random_state=0,compute_importances=True)
tree.fit(TrainX,TrainY)
treeOut = tree.predict(TestX)
print sum(treeOut == TestY)/float(len(treeOut))

forest1 = RandomForestClassifier(n_estimators=50, max_depth=None,
                                 min_samples_split=1, random_state=0,
                                compute_importances=True)
forest1.fit(TrainX,TrainY)
forestOut1 = forest1.predict(TestX)                             
print sum(forestOut1 == TestY)/float(len(forestOut1))
                                                                                          
forest2 = ExtraTreesClassifier(n_estimators=50, max_depth=None,
                               min_samples_split=1, random_state=0,
                                compute_importances=True)
forest2.fit(TrainX,TrainY)
forestOut2 = forest2.predict(TestX)                             
print sum(forestOut2 == TestY)/float(len(forestOut2))

forest3 = AdaBoostClassifier(n_estimators=50, random_state=0)
forest3.fit(TrainX,TrainY)
forestOut3 = forest3.predict(TestX)                             
print sum(forestOut3 == TestY)/float(len(forestOut3))

print sum((treeOut != TestY) & (forestOut1 != TestY) & (forestOut2 != TestY) & (forestOut3 != TestY))

#most important features in each classifier
def ImpFeatures(forest,feature_list):
    df= DataFrame()
    df["importance"]=forest.feature_importances_
    df.sort(columns="importance", inplace=True,ascending=False)
    df["features"]= feature_list[df.index]
    return df

t_df= ImpFeatures(tree,Food.columns)
f1_df= ImpFeatures(forest1,Food.columns)
f2_df= ImpFeatures(forest2,Food.columns)
#AdaBoostClassifier not have: compute_importances??
print "tree\n",t_df.head()
print "forest\n", f1_df.head()
print "forest2\n",f2_df.head()