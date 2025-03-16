
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt

data1 = pd.read_excel("U.xlsx")
data2 = pd.read_excel("V.xlsx")

def grade_assign(marks):
  if(marks>90):
    return 10
  elif(marks>85):
    return 9
  elif(marks>80):
    return 8
  elif(marks>70):
    return 7
  elif(marks>60):
    return 6
  elif(marks>50):
    return 5
  else:
    return 4

da=data1[data1['Paper_ID']=="SEMI0052593"]#M5
pre=[]
for i in range(len(da)):
  m=da.iloc[i,2]
  pre.append(grade_assign(m))

ccmse=[]
spmse=[]
dpmse=[]
sdmse=[]

students=da.iloc[:,0]

subjects=data2.loc[data2.Mathematics>0,:]

p2=[]
p3=[]
p4=[]
p5=[]
pf=[]
data3=subjects.iloc(axis=0)[0:7]

st=np.zeros(10000)

for j in students:
  stuid=j
  l=[]
  for i in range(len(data1)):
    ro=data1.iloc[i,1]
    ma=data1.iloc[i,2]
    stu=data1.iloc[i,0]

    if(stu==stuid):
      if(ro=='SEMI0015183'or ro=='SEMI0025909' or ro=='SEMI0037924'or ro=='SEMI0039951'or ro=='SEMI0044637'or ro=='SEMI0044275'or ro=='SEMI0052593'):
        l.append(ma)

    me=0
    for i in l:
      me+=i
    me=me/7
    avg=[]
    for i in range(7):
      avg.append(me)
  if(len(l)<7):
     for k in range(7-len(l)):
        l.append(50)
  data3=data3.iloc(axis=0)[0:7]
  data4=pd.DataFrame(data3)
  #data3=data3[['Marks']]
  data4['avg']=avg
  data4['me_marks']=l;
  data4=data3.iloc(axis=1)[17:]
  predict="me_marks"
  p=data4[predict]
  X = np.array(data4.drop([predict], 1))
  y = np.array(data4[predict])
  kfold = sklearn.model_selection.KFold(n_splits=3, random_state=10,shuffle=True)
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.10, random_state=40,shuffle=True)
  w1=0.1
  w2=0.2
  w3=0.3
  w4=0.4

  model1 = tree.DecisionTreeClassifier()
  model2 = KNeighborsClassifier()
  model3= LogisticRegression()
  model4=linear_model.LinearRegression(fit_intercept=True)
  model5=RandomForestRegressor()
  model1.fit(x_train,y_train)
  model2.fit(x_train,y_train)
  model3.fit(x_train,y_train)
  model4.fit(x_train,y_train)
  model5.fit(x_train,y_train);

  pred1=model1.predict_proba(x_test)
  pred2=model2.predict(x_test)
  pred3=model3.predict(x_test)
  pred4=model4.predict(x_test)
  pred5=model5.predict(x_test)

  finalpred=(pred4*0.5+pred2*0.3+pred3*0.2+pred5*0.3)

#COURSE CLUSTERING
pf=[]
subjects=data2.loc[data2.Mathematics>0,:]
subjects=subjects.iloc(axis=0)[0:6]
data3=subjects.iloc(axis=0)[0:2]
st=np.zeros(10000)
for j in students:
  stuid=j
  l=[]
  for i in range(len(data1)):
    ro=data1.iloc[i,1]
    ma=data1.iloc[i,2]
    stu=data1.iloc[i,0]

    if(stu==stuid):
      if(ro=='SEMI0015183'or ro=='SEMI0025909'):
        l.append(ma)
    me=0
    for i in l:
      me+=i
    me=me/2
    avg=[]
    for i in range(2):
      avg.append(me)
  if(len(l)<2):
     for k in range(2-len(l)):
        l.append(50)

  data3=data3.iloc(axis=0)[0:2]
  data4=pd.DataFrame(data3)
  data4['avg']=avg
  data4['me_marks']=l;
  data4=data3.iloc(axis=1)[17:]
  predict="me_marks"
  p=data4[predict]
  X = np.array(data4.drop([predict], 1))
  y = np.array(data4[predict])
  kfold = sklearn.model_selection.KFold(n_splits=3, random_state=10,shuffle=True)
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.10, random_state=40,shuffle=True)
  w1=0.1
  w2=0.2
  w3=0.3
  w4=0.4

  model2 = KNeighborsClassifier()
  model3= LogisticRegression()
  model4=linear_model.LinearRegression(fit_intercept=True)
  model5=RandomForestRegressor()

  model2.fit(x_train,y_train)
  model4.fit(x_train,y_train)
  model5.fit(x_train,y_train);

  pred4=model4.predict(x_test)
  pred5=model5.predict(x_test)

  finalpred=(pred4*0.5+pred2*0.1+pred3*0.1+pred5*0.3)

  pf.append(grade_assign(finalpred))
Y_true = pre
Y_pred = pf
temp=mean_squared_error(Y_true,Y_pred)
ccmse.append(temp)
pf=[]
data3=subjects.iloc(axis=0)[0:5]

st=np.zeros(10000)

for j in students:
  stuid=j
  l=[]
  for i in range(len(data1)):
    ro=data1.iloc[i,1]
    ma=data1.iloc[i,2]
    stu=data1.iloc[i,0]

    if(stu==stuid):
      if(ro=='SEMI0015183'or ro=='SEMI0025909' or ro=='SEMI0037924'or ro=='SEMI0039951'or ro=='SEMI0044637'):
        l.append(ma)
    me=0
    for i in l:
      me+=i
    me=me/5
    avg=[]
    for i in range(5):
      avg.append(me)
  if(len(l)<5):
     for k in range(5-len(l)):
        l.append(50)

  data3=data3.iloc(axis=0)[0:5]
  data4=pd.DataFrame(data3)
  data4['avg']=avg
  data4['me_marks']=l;
  data4=data3.iloc(axis=1)[17:]
  predict="me_marks"
  p=data4[predict]
  X = np.array(data4.drop([predict], 1))
  y = np.array(data4[predict])
  kfold = sklearn.model_selection.KFold(n_splits=3, random_state=10,shuffle=True)
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.10, random_state=40,shuffle=True)
  w1=0.1
  w2=0.2
  w3=0.3
  w4=0.4

  model2 = KNeighborsClassifier()
  model3= LogisticRegression()
  model4=linear_model.LinearRegression(fit_intercept=True)
  model5=RandomForestRegressor()

  model2.fit(x_train,y_train)
  model3.fit(x_train,y_train)
  model4.fit(x_train,y_train)
  model5.fit(x_train,y_train);

  pred3=model3.predict(x_test)
  pred4=model4.predict(x_test)
  pred5=model5.predict(x_test)

  finalpred=(pred4*0.5+pred2*0.1+pred3*0.1+pred5*0.3)
  pf.append(grade_assign(finalpred))
Y_true = pre
Y_pred = pf
if (temp<mean_squared_error(Y_true,Y_pred)):
  ccmse.append(temp)
else:
  temp=mean_squared_error(Y_true,Y_pred)
  ccmse.append(temp)
students=da.iloc[:,0]
pf=[]
subjects=data2.loc[data2.Mathematics>0,:]
subjects=subjects.iloc(axis=0)[0:6]

data3=subjects.iloc(axis=0)[0:6]

st=np.zeros(10000)

for j in students:
  stuid=j
  l=[]
  for i in range(len(data1)):
    ro=data1.iloc[i,1]
    ma=data1.iloc[i,2]
    stu=data1.iloc[i,0]

    if(stu==stuid):
      if(ro=='SEMI0015183'or ro=='SEMI0025909' or ro=='SEMI0037924'or ro=='SEMI0039951'or ro=='SEMI0044637'or ro=='SEMI0044275'):
        l.append(ma)
        #print(ma)
    me=0
    for i in l:
      me+=i
    me=me/6
    avg=[]
    for i in range(6):
      avg.append(me)
  if(len(l)<6):
     for k in range(6-len(l)):
        l.append(50)
  data3=data3.iloc(axis=0)[0:6]
  data4=pd.DataFrame(data3)
  data4['avg']=avg
  data4['me_marks']=l;
  data4=data3.iloc(axis=1)[17:]
  predict="me_marks"
  p=data4[predict]
  X = np.array(data4.drop([predict], 1))
  y = np.array(data4[predict])
  kfold = sklearn.model_selection.KFold(n_splits=3, random_state=10,shuffle=True)
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.10, random_state=40,shuffle=True)
  w1=0.1
  w2=0.2
  w3=0.3
  w4=0.4


  model2 = KNeighborsClassifier()
  model3= LogisticRegression()
  model4=linear_model.LinearRegression(fit_intercept=True)
  model5=RandomForestRegressor()

  model2.fit(x_train,y_train)
  model3.fit(x_train,y_train)
  model4.fit(x_train,y_train)
  model5.fit(x_train,y_train);


  pred2=model2.predict(x_test)
  pred3=model3.predict(x_test)
  pred4=model4.predict(x_test)
  pred5=model5.predict(x_test)

  finalpred=(pred4*0.5+pred2*0.1+pred3*0.1+pred5*0.3)
  pf.append(grade_assign(finalpred))
Y_true = pre
Y_pred = pf
if (temp<mean_squared_error(Y_true,Y_pred)):
  ccmse.append(temp)
else:
  temp=mean_squared_error(Y_true,Y_pred)
  ccmse.append(temp)
#4th yr
students=da.iloc[:,0]

subjects=data2.loc[data2.Mathematics>0,:]

pf=[]
data3=subjects.iloc(axis=0)[0:7]
st=np.zeros(10000)

for j in students:
  stuid=j
  l=[]
  for i in range(len(data1)):
    ro=data1.iloc[i,1]
    ma=data1.iloc[i,2]
    stu=data1.iloc[i,0]

    if(stu==stuid):
      if(ro=='SEMI0015183'or ro=='SEMI0025909' or ro=='SEMI0037924'or ro=='SEMI0039951'or ro=='SEMI0044637'or ro=='SEMI0044275'or ro=='SEMI0052593'):
        l.append(ma)

    me=0
    for i in l:
      me+=i
    me=me/7
    avg=[]
    for i in range(7):
      avg.append(me)
  if(len(l)<7):
     for k in range(7-len(l)):
        l.append(50)
  data3=data3.iloc(axis=0)[0:7]
  data4=pd.DataFrame(data3)
  data4['avg']=avg
  data4['me_marks']=l;
  data4=data3.iloc(axis=1)[17:]
  predict="me_marks"
  p=data4[predict]
  X = np.array(data4.drop([predict], 1))
  y = np.array(data4[predict])
  kfold = sklearn.model_selection.KFold(n_splits=3, random_state=10,shuffle=True)
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.10, random_state=40,shuffle=True)
  w1=0.1
  w2=0.2
  w3=0.3
  w4=0.4


  model2 = KNeighborsClassifier()
  model3= LogisticRegression()
  model4=linear_model.LinearRegression(fit_intercept=True)
  model5=RandomForestRegressor()

  model2.fit(x_train,y_train)
  model3.fit(x_train,y_train)
  model4.fit(x_train,y_train)
  model5.fit(x_train,y_train);


  pred2=model2.predict(x_test)
  pred3=model3.predict(x_test)
  pred4=model4.predict(x_test)
  pred5=model5.predict(x_test)

  finalpred=(pred4*0.5+pred2*0.1+pred3*0.1+pred5*0.3)
  pf.append(grade_assign(finalpred))
Y_true = pre
Y_pred = pf
if (temp<mean_squared_error(Y_true,Y_pred)):
  ccmse.append(temp)
else:
  temp=mean_squared_error(Y_true,Y_pred)
  ccmse.append(temp)

ccmse



#Direct prerequisite
pf=[]
subjects=data2.loc[data2.Mathematics>0,:]
subjects=subjects.iloc(axis=0)[0:6]
data3=subjects.iloc(axis=0)[0:2]
st=np.zeros(10000)
for j in students:
  stuid=j
  l=[]
  for i in range(len(data1)):
    ro=data1.iloc[i,1]
    ma=data1.iloc[i,2]
    stu=data1.iloc[i,0]

    if(stu==stuid):
      if(ro=='SEMI0044637'or ro=='SEMI0044275'):
        l.append(ma)
    me=0
    for i in l:
      me+=i
    me=me/2
    avg=[]
    for i in range(2):
      avg.append(me)
  if(len(l)<2):
     for k in range(2-len(l)):
        l.append(50)

  data3=data3.iloc(axis=0)[0:2]
  data4=pd.DataFrame(data3)
  data4['avg']=avg
  data4['me_marks']=l;
  data4=data3.iloc(axis=1)[17:]
  predict="me_marks"
  p=data4[predict]
  X = np.array(data4.drop([predict], 1))
  y = np.array(data4[predict])
  kfold = sklearn.model_selection.KFold(n_splits=3, random_state=10,shuffle=True)
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.10, random_state=40,shuffle=True)
  w1=0.1
  w2=0.2
  w3=0.3
  w4=0.4

  model2 = KNeighborsClassifier()
  model3= LogisticRegression()
  model4=linear_model.LinearRegression(fit_intercept=True)
  model5=RandomForestRegressor()

  model2.fit(x_train,y_train)
  model4.fit(x_train,y_train)
  model5.fit(x_train,y_train);

  pred4=model4.predict(x_test)
  pred5=model5.predict(x_test)

  finalpred=(pred4*0.5+pred2*0.1+pred3*0.15+pred5*0.3)

  pf.append(grade_assign(finalpred))
Y_true = pre
Y_pred = pf
temp=mean_squared_error(Y_true,Y_pred)
dpmse.append(temp)
pf=[]
data3=subjects.iloc(axis=0)[0:5]

st=np.zeros(10000)

for j in students:
  stuid=j
  l=[]
  for i in range(len(data1)):
    ro=data1.iloc[i,1]
    ma=data1.iloc[i,2]
    stu=data1.iloc[i,0]

    if(stu==stuid):
      if(ro=='SEMI0044637'or ro=='SEMI0044275'):
        l.append(ma)
    me=0
    for i in l:
      me+=i
    me=me/5
    avg=[]
    for i in range(5):
      avg.append(me)
  if(len(l)<5):
     for k in range(5-len(l)):
        l.append(50)

  data3=data3.iloc(axis=0)[0:5]
  data4=pd.DataFrame(data3)
  data4['avg']=avg
  data4['me_marks']=l;
  data4=data3.iloc(axis=1)[17:]
  predict="me_marks"
  p=data4[predict]
  X = np.array(data4.drop([predict], 1))
  y = np.array(data4[predict])
  kfold = sklearn.model_selection.KFold(n_splits=3, random_state=10,shuffle=True)
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.10, random_state=40,shuffle=True)
  w1=0.1
  w2=0.2
  w3=0.3
  w4=0.4

  model2 = KNeighborsClassifier()
  model3= LogisticRegression()
  model4=linear_model.LinearRegression(fit_intercept=True)
  model5=RandomForestRegressor()

  model2.fit(x_train,y_train)
  model3.fit(x_train,y_train)
  model4.fit(x_train,y_train)
  model5.fit(x_train,y_train);

  pred3=model3.predict(x_test)
  pred4=model4.predict(x_test)
  pred5=model5.predict(x_test)

  finalpred=(pred4*0.5+pred2*0.15+pred3*0.1+pred5*0.3)
  pf.append(grade_assign(finalpred))
Y_true = pre
Y_pred = pf
if (temp<mean_squared_error(Y_true,Y_pred)):
  dpmse.append(temp)
else:
  temp=mean_squared_error(Y_true,Y_pred)
  ccmse.append(temp)
students=da.iloc[:,0]
pf=[]
subjects=data2.loc[data2.Mathematics>0,:]
subjects=subjects.iloc(axis=0)[0:6]

data3=subjects.iloc(axis=0)[0:6]

st=np.zeros(10000)

for j in students:
  stuid=j
  l=[]
  for i in range(len(data1)):
    ro=data1.iloc[i,1]
    ma=data1.iloc[i,2]
    stu=data1.iloc[i,0]

    if(stu==stuid):
      if(ro=='SEMI0044637'or ro=='SEMI0044275'):
        l.append(ma)
        #print(ma)
    me=0
    for i in l:
      me+=i
    me=me/6
    avg=[]
    for i in range(6):
      avg.append(me)
  if(len(l)<6):
     for k in range(6-len(l)):
        l.append(50)
  data3=data3.iloc(axis=0)[0:6]
  data4=pd.DataFrame(data3)
  data4['avg']=avg
  data4['me_marks']=l;
  data4=data3.iloc(axis=1)[17:]
  predict="me_marks"
  p=data4[predict]
  X = np.array(data4.drop([predict], 1))
  y = np.array(data4[predict])
  kfold = sklearn.model_selection.KFold(n_splits=3, random_state=10,shuffle=True)
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.10, random_state=40,shuffle=True)
  w1=0.1
  w2=0.2
  w3=0.3
  w4=0.4


  model2 = KNeighborsClassifier()
  model3= LogisticRegression()
  model4=linear_model.LinearRegression(fit_intercept=True)
  model5=RandomForestRegressor()

  model2.fit(x_train,y_train)
  model3.fit(x_train,y_train)
  model4.fit(x_train,y_train)
  model5.fit(x_train,y_train);


  pred2=model2.predict(x_test)
  pred3=model3.predict(x_test)
  pred4=model4.predict(x_test)
  pred5=model5.predict(x_test)

  finalpred=(pred4*0.5+pred2*0.2+pred3*0.1+pred5*0.3)
  pf.append(grade_assign(finalpred))
Y_true = pre
Y_pred = pf
if (temp<mean_squared_error(Y_true,Y_pred)):
  dpmse.append(temp)
else:
  temp=mean_squared_error(Y_true,Y_pred)
  ccmse.append(temp)
#4th yr
students=da.iloc[:,0]

subjects=data2.loc[data2.Mathematics>0,:]

pf=[]
data3=subjects.iloc(axis=0)[0:7]
st=np.zeros(10000)

for j in students:
  stuid=j
  l=[]
  for i in range(len(data1)):
    ro=data1.iloc[i,1]
    ma=data1.iloc[i,2]
    stu=data1.iloc[i,0]

    if(stu==stuid):
      if(ro=='SEMI0044637'or ro=='SEMI0044275'):
        l.append(ma)

    me=0
    for i in l:
      me+=i
    me=me/7
    avg=[]
    for i in range(7):
      avg.append(me)
  if(len(l)<7):
     for k in range(7-len(l)):
        l.append(50)
  data3=data3.iloc(axis=0)[0:7]
  data4=pd.DataFrame(data3)
  data4['avg']=avg
  data4['me_marks']=l;
  data4=data3.iloc(axis=1)[17:]
  predict="me_marks"
  p=data4[predict]
  X = np.array(data4.drop([predict], 1))
  y = np.array(data4[predict])
  kfold = sklearn.model_selection.KFold(n_splits=3, random_state=10,shuffle=True)
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.10, random_state=40,shuffle=True)
  w1=0.1
  w2=0.2
  w3=0.3
  w4=0.4


  model2 = KNeighborsClassifier()
  model3= LogisticRegression()
  model4=linear_model.LinearRegression(fit_intercept=True)
  model5=RandomForestRegressor()

  model2.fit(x_train,y_train)
  model3.fit(x_train,y_train)
  model4.fit(x_train,y_train)
  model5.fit(x_train,y_train);


  pred2=model2.predict(x_test)
  pred3=model3.predict(x_test)
  pred4=model4.predict(x_test)
  pred5=model5.predict(x_test)

  finalpred=(pred4*0.5+pred2*0.1+pred3*0.1+pred5*0.3)
  pf.append(grade_assign(finalpred))
Y_true = pre
Y_pred = pf
if (temp<mean_squared_error(Y_true,Y_pred)):
  dpmse.append(temp)
else:
  temp=mean_squared_error(Y_true,Y_pred)
  dpmse.append(temp)

plt.plot(ccmse,color='black',label='Course clustering')
plt.plot(dpmse,color='orange',label='Direct Prerequiste')
plt.ylabel('MeanSquareError')
plt.xlabel('Time(Term)')
plt.legend()
