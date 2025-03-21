
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pylab as plt

def grade_assignment(marks):
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

data=pd.read_excel("U.xlsx")
stud=data.Student_ID.unique()
stud

data

st=np.zeros(10000)
c=np.zeros(10000)
for i in range(len(data)):
  mark=data.iloc[i,2]
  stu=data.iloc[i,0]
  st[stu]+=grade_assignment(mark)
  c[stu]+=1
for i in stud:
  st[i]/=c[i]
g=[]
for i in stud:
  g.append(st[i])

#getting high school gpa and school gpa
hsgpa=np.zeros(10000)
sgpa=np.zeros(10000)
hs=[]
s=[]

for i in range(len(data)):
  stu=data.iloc[i,0]
  hsgpa[stu]=data.iloc[i,4]
  sgpa[stu]=data.iloc[i,3]

for i in stud:
  hs.append(hsgpa[i])
  s.append(sgpa[i])

X = np.array(s)
y = np.array(g)
plt.plot(X, y, 'o')
m, b = np.polyfit(X, y, 1)
plt.plot(X, m*X + b)
plt.title('Correlation between Final GPAs VS School GPAs')
plt.ylabel("Final GPA")
plt.xlabel("School GPA")

X = np.array(hs)
y = np.array(g)
plt.plot(X, y, 'o')
m, b = np.polyfit(X, y, 1)
plt.plot(X, m*X + b)
plt.title('Correlation between Final GPAs VS 12th GPAs')
plt.ylabel("Final GPA")
plt.xlabel("High School GPA")

