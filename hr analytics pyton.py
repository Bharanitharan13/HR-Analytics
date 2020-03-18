import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt 

#setting path
os.chdir('E:\\hr analytics')

train = pd.read_csv('train_LZdllcl.csv')
test = pd.read_csv('test_2umaH9m.csv')

missing_value = train.isnull().sum()

train['education'].value_counts()


mode_education = train['education'].mode()

mode_education[0]

train['education'] = train['education'].fillna(mode_education[0])

train['education'].isnull().sum()

for col in train.columns:
    print(col)
    
train['previous_year_rating'].plot.hist()

train['previous_year_rating'].value_counts().plot.bar()


sns.pairplot(train)

coor = train.corr()

train['previous_year_rating'] = train['previous_year_rating'].fillna(1)

train['previous_year_rating'].isnull().sum()

train_missing_updated = train.isnull().sum()

table = pd.crosstab(train['education'],train['gender'])

from scipy.stats import chi2_contingency

chiqex = chi2_contingency(table)
tbl=pd.crosstab(train['education'],train['gender'])
chiq1=chi2_contingency(tbl)
tbl1=pd.crosstab(train['department'],train['is_promoted'])
chiq2=chi2_contingency(tbl1)
tbl2=pd.crosstab(train['region'],train['is_promoted'])
chiq3=chi2_contingency(tbl2)
tbl3=pd.crosstab(train['education'],train['is_promoted'])
chiq4=chi2_contingency(tbl3)
tbl4=pd.crosstab(train['KPIs_met >80%'],train['is_promoted'])
chiq5=chi2_contingency(tbl4)
tbl5=pd.crosstab(train['previous_year_rating'],train['is_promoted'])
chiq6=chi2_contingency(tbl5)

tbl.plot.bar()
tbl1.plot.bar()
tbl2.plot.bar()
tbl3.plot.bar()

train.skew()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['department']=le.fit_transform(train['department'])

train['department'].value_counts()
plt.figure(figsize=(30,30))
sns.heatmap(train.corr(),cmap='coolwarm',annot = True,annot_kws={'size':15})

categorical_feature_mask = train.dtypes==object
train.columns[categorical_feature_mask].tolist()


train['region']=le.fit_transform(train['region'])

train['education']=le.fit_transform(train['education'])

train['gender']=le.fit_transform(train['gender'])

train['recruitment_channel']=le.fit_transform(train['recruitment_channel'])


y = train['is_promoted']

train['is_promoted'].value_counts()

x = train.iloc[:,1:13]

from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=123)

import statsmodels.api as sn

logit = sn.Logit(y_train,x_train)
result = logit.fit()

result.summary()

preds = result.predict(x_train)

preds= np.where(preds>0.5,1,0)
from sklearn.metrics import confusion_matrix

cof_mat = confusion_matrix(y_train,preds)

from sklearn.metrics import f1_score

### calculating f1score

f1_score(y_train,preds)

preds_ytest = result.predict(x_test)
preds_ytest = np.where(preds_ytest>0.5,1,0)

f1_score(y_test,preds_ytest)

##### decision treee
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()

DTC.fit(x_train,y_train)

pred_dtc = DTC.predict(x_train)

f1_score(y_train,pred_dtc)

###validation data set

#pred_dtc = DTC.predict(test)

categorical_feature_mask = test.dtypes==object
cat = test.columns[categorical_feature_mask].tolist()
cat

def categorical_variable(dataframe):
    variable_name=[i for i in dataframe.columns if dataframe.dtypes[i]=='object']
    for x in variable_name:
        dataframe[x]=le.fit_transform(dataframe[x])
    return dataframe

categorical_variable(test)

##  supoort vector machine

from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train,y_train)
preds_svm = svm.predict(x_train)
cm_svm = confusion_matrix(y_train,preds_svm)
f1score_svm = f1_score(y_train,preds_svm)
f1score_svm

preds_svm_train = svm.predict(x_test)
f1score_svm_train = f1_score(y_test,preds_svm_train)
f1score_svm_train

test.isna().sum()


