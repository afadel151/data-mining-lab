#%% 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
plt.style.use('ggplot')
print("imported")

df = pd.read_csv("titanic3.csv")

df = df.drop(["name", "sibsp", "parch","sex", "ticket", "cabin", "embarked","boat", "body", "home.dest"]
				, axis=1)


df = df.dropna(axis=0)

y = df['survived']

X = df.drop('survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2,random_state=5)
 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#%% 
model = KNeighborsClassifier(n_neighbors=8)
model.fit(X_train,y_train)
print(model.score(X_train,y_train),model.score(X_test,y_test))

#%%

from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_auc_score,roc_curve,classification_report

#%% 
y_pred = model.predict(X_test)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("Precision: ",precision_score(y_test, y_pred))
print("F1: ",f1_score(y_test, y_pred))
print("Recall: ",recall_score(y_test, y_pred))

#%%
y_score= model.predict_proba(X_test)[:,0]
print(roc_auc_score(y_test, y_score))
fpr, tpr, _= roc_curve(y_test, y_score)
plt.plot(tpr,fpr)
plt.plot([0,1],[0,1])

#%% 
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, model.predict(X_test)))

#%% 

cm = confusion_matrix(y_test, y_pred)	
print(classification_report(y_test, y_pred))
print(f'ROC AUC score: {roc_auc_score(y_test, y_score)}')
print('Accuracy Score: ',accuracy_score(y_test, y_pred))
plt.figure(figsize = (8, 5))
sns.heatmap(cm,
 	cmap = 'Blues',
 	annot = True,
 	fmt = 'd',
 	linewidths = 5,
 	cbar =False,
 	annot_kws = {'fontsize': 15},
	yticklabels = ['No survived', 'survived'], 
	xticklabels =['Predicted no survived','Prediction survived']
)
plt.yticks(rotation=0)
plt.show()

#%% 
# ROC CURVE 
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_score)
roc_auc = roc_auc_score(y_test, y_score)
sns.set_theme(style = 'white')
plt.figure(figsize = (8, 8))
plt.plot(
	false_positive_rate,
	true_positive_rate, 
	color = '#b01717', 
	label ='AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


#%% 

from sklearn.model_selection import cross_val_score

#%% 

print(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))

#%% 

from sklearn.model_selection import KFold
cv= KFold()

print(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'))

#%% 
from sklearn.model_selection import ShuffleSplit

cv= ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
print(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'))

#%% 

from sklearn.model_selection import LeaveOneOut
cv= LeaveOneOut()
print(cross_val_score(model,X_train,y_train,cv=cv,scoring='accuracy'))

#%% 
from sklearn.model_selection import StratifiedKFold
cv= StratifiedKFold(n_splits=5, shuffle=True)
print(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'))

#%% 
val_score = []
for k in range(1, 50):
	score = cross_val_score(
		KNeighborsClassifier(n_neighbors=k), 
		X_train,
		y_train, 
		cv=5).mean()
	val_score.append(score)

plt.plot(val_score)
plt.ylabel('score')
plt.xlabel('n_neighbors')


#%% 
from sklearn.model_selection import validation_curv

k = np.arange(1, 50)
train_score,val_score=validation_curve(model, 
	X_train,
	y_train,
	param_name='n_neighbors', 
	param_range=k, 
	cv=10)
plt.plot(k, val_score.mean(axis=1), label='validation')
plt.plot(k, train_score.mean(axis=1), label='train')
plt.ylabel('score')
plt.xlabel('n_neighbors')
plt.legend()
