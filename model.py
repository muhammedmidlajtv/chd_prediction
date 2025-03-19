

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df= pd.read_csv("framingham.csv")

sns.countplot(x='TenYearCHD', data=df)
sns.countplot(x= 'male', data= df)

sns.heatmap(df.corr(),annot= True)
df1= df.drop(['education', 'currentSmoker'], axis='columns')
df1.head()
df1.isna().sum()

df1= df1.fillna({
    'BPMeds': 0,
    'cigsPerDay': 0
})
df1.isna().sum()

missing= df[df['totChol'].isna()==True]
print(missing)

df['totChol'][42]

df1=df1.fillna({
    'totChol': df1['totChol'].mean()
})
df1.isna().sum()
df1['totChol'][42]

df1=df1.fillna({
    'glucose': df1['glucose'].mean(),
    'BMI': df1['BMI'].mean(),
    'heartRate': df1['heartRate'].mean()
})

df1.isna().sum()

df1.describe()

sns.boxplot(df1['totChol'])

df2= df1.drop(df1[df1['totChol']>500].index)            
sns.boxplot(df2['totChol'])

df2.describe()

df2[df2['sysBP'] > df2['sysBP'].mean() + 3*(df2['sysBP'].std())]

df3= df2[df2['sysBP']<= df2['sysBP'].mean() + 3* (df2['sysBP'].std())]
df3.shape

df3[df3['diaBP']> df3['diaBP'].mean() + 3*( df3['diaBP'].std())]

df4= df3[df3['diaBP'] <= df3['diaBP'].mean() + 3*(df3['diaBP'].std())]
df4.shape

df4[df4['BMI']> df4['BMI'].mean() + 3*(df4['BMI'].std())]

df5= df4[df4['BMI'] <= df4['BMI'].mean() + 3*(df4['BMI'].std())]
df5.shape
df5[df5['heartRate'] > df5['heartRate'].mean() + 3*(df5['heartRate'].std())]

df6= df5[df5['heartRate'] <= df5['heartRate'].mean() + 3*(df5['heartRate'].std())]
df6.shape

df6[df6['glucose']> df6['glucose'].mean() + 3*(df6['glucose'].std())]

df7= df6[df6['glucose'] <= df6['glucose'].mean() + 3*(df6['glucose'].std())]
df7.shape

x= df7.drop(['TenYearCHD'], axis= 'columns')
x.head()

y=df7['TenYearCHD']
y.head()


from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()
x_scaled= scaler.fit_transform(x)
x_scaled
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x_scaled, y, test_size=0.2,stratify=y, random_state= 20)
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression(solver= 'liblinear')
lr.fit(x_train, y_train)
lr.score(x_train, y_train)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
model_params= {
    'logistic_reg':{
        'model': LogisticRegression(solver= 'liblinear'),
        'params': {
            'C':[1,5,10]
        }
    },

    'svm':{
        'model': SVC(gamma= 'auto'),
        'params': {
            'kernel': ['rbf', 'linear'],
            'C': [1,5,10,20]
        }
    },

    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy']
        }
    },

    'random_forest':{
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [5,10,20]
        }
    },

    'gaussian_nb':{
        'model': GaussianNB(),
        'params': {}
    }
}
from sklearn.model_selection import GridSearchCV

score=[]

for model_name, mp in model_params.items():
  gs= GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
  gs.fit(x_scaled, y)
  score.append({
      'model_name': model_name,
      'best_score': gs.best_score_,
      'best_params': gs.best_params_
  })

df_score= pd.DataFrame(score, columns=['model_name', 'best_score', 'best_params'])
df_score
from sklearn.decomposition import PCA

pca= PCA(0.95)
x_pca= pca.fit_transform(x_scaled)
x_pca.shape
x_train_pca, x_test_pca, y_train, y_test= train_test_split(x_pca, y, test_size= 0.2 ,random_state=20)
lr= LogisticRegression(solver='liblinear', C=10)
lr.fit(x_train_pca, y_train)
lr.score(x_test_pca, y_test)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x_scaled, y, test_size=0.2,stratify=y, random_state= 20)
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression(solver= 'liblinear')
lr.fit(x_train, y_train)
lr.score(x_train, y_train)

joblib.dump(lr, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")

from sklearn.metrics import confusion_matrix
y_predicted= lr.predict(x_test)
cm=confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True)

