#Imported Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

get_ipython().magic('matplotlib inline')


#Loading Data
train = pd.read_csv('train.csv')
label = pd.DataFrame(train.label)
train = train[train.columns.drop('label')]
test = pd.read_csv('test.csv')

trainImages, trainLabels = train, label
testImages, testLabels = test, label


#Put the pixels into an image
i = 0
num = Image.new("RGB",(28, 28))
px = num.load()
for y in range(28):
    for x in range(28):
        px[x,y] = (train.iloc[500,i],0,0)
        i = i + 1
        
num


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics

y = label
X = train

error_rate = []

cv = cross_validation.KFold(len(X), n_folds=5, shuffle=True)

for train_index, test_index in cv:
    model = RandomForestClassifier(n_estimators=75, max_depth=15, min_samples_leaf=6).fit(X.iloc[train_index], y.iloc[train_index])
    df = y.iloc[test_index]
    df['predict'] = model.predict(X.iloc[test_index])
    error_rate.append(float(len(df[df.label != df.predict]))/float(len(df)))
    
    
print("Error Rate:", np.mean(error_rate))



test['label'] = pd.DataFrame(model.predict(test.fillna(0)))
pd.DataFrame(test.label)



#Map pixels into an image
from PIL import Image

i = 0
num = Image.new("RGB",(28, 28))
px = num.load()
for y in range(28):
    for x in range(28):
        px[x,y] = (test.iloc[9,i],0,0)
        i = i + 1        
num




# Plot pixel importances
importances = model.feature_importances_
importances = importances.reshape((28, 28))
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances for random forests")
plt.show()



#Cross Validation Score
scores = cross_validation.cross_val_score(model, train[:1000], train[:1000].values.tolist(), cv=5)
print (scores)

