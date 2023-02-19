#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[2]:



data=pd.read_csv("https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv")
x=data.drop('Class',axis=1)
y=data['Class']
print("no of rows belonging to class 0:",(y==0).sum()) # 0 for not fraud.
print("no of rows belonging to class 1:",(y==1).sum())


# In[3]:


plot = data['Class'].value_counts().plot(figsize=(5,5),kind='bar',colormap='Accent')


# In[4]:



ros = RandomOverSampler(random_state=42)
x_ros, y_ros = ros.fit_resample(x, y)
df=pd.DataFrame(x_ros)
df['Class']=y_ros
plot = df['Class'].value_counts().plot(figsize=(5,5),kind='bar',colormap='Accent')


# In[5]:


z=1.645 #z score for 90%  confidence
p=0.5 #assumed to be 0.5 mostly
E=0.10 # 10% margin of erro
sampleSize = math.ceil((z*z*p*(1-p))/(E*E))


final_samples=[]
randomsampling = df.sample(n=sampleSize, random_state=0)
final_samples.append(randomsampling)
n = len(df)
k = int(math.sqrt(n))
systematic = df.iloc[::k]
final_samples.append(systematic)
C=1.5
sample_size = round(((z**2)*p*(1-p))/((E/C)**2))
no_of_clusters=2
df_new=df
N = len(df)
K = int(N/sampleSize)
data = None
for k in range(K):
    sample_k = df_new.sample(sampleSize)
    sample_k["cluster"] = np.repeat(k,len(sample_k))
    df_new = df_new.drop(index = sample_k.index)
    data = pd.concat([data,sample_k],axis = 0)

random_chosen_clusters = np.random.randint(0,K,size = no_of_clusters)
cluster = data[data.cluster.isin(random_chosen_clusters)]
cluster.drop(['cluster'], axis=1, inplace=True)
final_samples.append(cluster)

stratified=df.groupby('Class', group_keys=False).apply(lambda x: x.sample(190))
final_samples.append(stratified)
conv=df.head(400)
final_samples.append(conv)



# In[6]:


methods=['Simple Random','Systematic','Cluster','Stratified','Convenience']
res=pd.DataFrame(columns=methods, index=['Logistic Regression','Naive Bayes','KNN','Decision Tree','Random Forest'])


# In[7]:



# Applying Models
for i in range(5):
    j=0
    x_s=final_samples[i].drop('Class',axis=1)
    y_s=final_samples[i]['Class']

    
    x_train, x_test, y_train, y_test = train_test_split(x_s ,y_s , random_state=104, test_size=0.25, shuffle=True)

    # Logistic Regression
    classifier = LogisticRegression(random_state = 0,max_iter=2000)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy= accuracy_score(y_test, y_pred)
    res.iloc[j,i]=accuracy
  

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    accuracy= accuracy_score(y_test, y_pred)
    res.iloc[j+1,i]=accuracy
    

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(x_train, y_train)
    y_pred=knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    res.iloc[j+2,i]=accuracy

    # Decision Tree
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5)
    clf_entropy.fit(x_train, y_train)
    y_pred=clf_entropy.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    res.iloc[j+3,i]=accuracy
    # print("Decision")

    # RandomForest Classifier
    clf = RandomForestClassifier(n_estimators = 100) 
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    res.iloc[j+4,i]=accuracy
print(res)


# In[ ]:




