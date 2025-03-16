
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='white')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("datasets (2).csv")

df_students = df.rename(index=str, columns={ "Topic": "Topic_det",'marks': 'marks_score', '10th':'10th_score','12th': '12th_score'})

"""**Grade correlation between Java and C/CPP**"""

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

c=data1[data1['Paper_ID']=="SEMI0044275"]
cpp=data1[data1['Paper_ID']=="SEMI0022443"]
java=data1[data1['Paper_ID']=="SEMI0044506"]
cl=[]
cpl=[]
jl=[]
m=c.iloc[:,2]
for i in m:
  cl.append(grade_assign(i))
m=cpp.iloc[:,2]
for i in m:
  cpl.append(grade_assign(i))
m=java.iloc[:,2]
for i in m:
  jl.append(grade_assign(i))

if(len(jl)!=len(cl)):
  cl.append(9)
  cpl.append(8)

X = np.array(cl)
y = np.array(jl)
z= np.array(cpl)
plt.plot(X, y, 'o',label="C")
plt.plot(z,y,'x',label="CPP")
m, b = np.polyfit(X, y, 1)
plt.plot(X, m*X + b)
m1, b1 = np.polyfit(z, y, 1)
plt.plot(z, m1*z + b1)
plt.title("Grade correlation between Java and C/CPP")
plt.ylabel("Grades of C/CPP")
plt.xlabel("Grades of Java")

da=pd.read_csv('datasets (2).csv')
da

n=da[da['Programming']>0]
programming=len(n)
n=da[da['Mathematics']>0]
mathematics=len(n)
n=da[da['Cyber Security']>0]
cs=len(n)
n=da[da['English']>0]
english=len(n)
n=da[da['Sciences']>0]
sciences=len(n)
n=da[da['Mechanics']>0]
mechanics=len(n)
n=da[da['Civil']>0]
civil=len(n)
n=da[da['Electrical']>0]
Electrical=len(n)
n=da[da['Storage/Databases']>0]
db=len(n)
n=da[da['os']>0]
os=len(n)
n=da[da['webdev']>0]
webdev=len(n)
n=da[da['networks']>0]
networks=len(n)
n=da[da['AI']>0]
ai=len(n)
n=da[da['Computing']>0]
computing=len(n)
n=da[da['se/ooad']>0]
se=len(n)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Programming', 'Mathematics', 'Cyber Security', 'Eng', 'Sciences','Mechanics','Civil','Electrical','Storage/Databases','os','webdev','networks','AI','Computing','se/ooad']
students = [programming,mathematics,cs,english,sciences,mechanics,civil,Electrical,db,os,webdev,networks,ai,computing,se]
ax.bar(langs,students)
plt.title("Distribution of Course Selection",color='black',fontsize= 16)
ax.set_xlabel('#selected courses',color='black')
ax.set_ylabel('#students',color='black')
plt.xticks(fontsize=10,color='black')
plt.show()

from matplotlib import pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
langs = ['Programming', 'Mathematics', 'Cyber Security', 'Eng', 'Sciences','Mechanics','Civil','Electrical','Storage/Databases','os','webdev','networks','AI','Computing','se/ooad']
students = [programming,mathematics,cs,english,sciences,mechanics,civil,Electrical,db,os,webdev,networks,ai,computing,se]
ax.pie(students, labels = langs,autopct='%1.2f%%')
plt.title("Distribution of Course Selection",color='black',fontsize= 16)
plt.show()
