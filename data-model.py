
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('clean.csv')
d = df
labels = df['rating_cat']
labels_cat = df["rating"]


df = df.drop(['rating'],axis=1)
data = df.drop(['rating_cat'],axis=1)

ind = np.arange(7) 
plt.scatter(data['headphone'],labels)
plt.xlabel('headphone')
plt.ylabel('Rating')
plt.xticks(ind,('3.5\nmm  ','2.5\nmm  ','USB-\nType C','Micro \nUSB','LUC','mini\n usb','MicroUSB \n+ 3.5mm'))
plt.plot()


x_train , x_test , y_train , y_test = train_test_split(data , labels , test_size = 0.33,random_state =90)
x_train_cat , x_test_cat , y_train_cat , y_test_cat = train_test_split(data , labels_cat , test_size = 0.33,random_state =90)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
score1 = cross_val_score(lm,x_train_cat,y_train_cat,cv=5)
print('Cross Validation Score : '+str(score1))
lm.fit(x_train_cat,y_train_cat)
print('LinearRegression : '+str(lm.score(x_test_cat,y_test_cat)*100))


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
score2 = cross_val_score(dtree,x_train,y_train,cv=5)
print('Cross Validation Score : '+str(score2))
dtree.fit(x_train,y_train)
print('DecisionTreeClassifier : '+str(dtree.score(x_test,y_test)*100))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
score3 = cross_val_score(rfc,x_train,y_train,cv=5)
print('Cross Validation Score : '+str(score3))
rfc.fit(x_train, y_train)
print('RandomForestClassifier : '+str(rfc.score(x_test,y_test)*100))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
score4 = cross_val_score(knn,x_train,y_train,cv=5)
print('Cross Validation Score : '+str(score4))
knn.fit(x_train,y_train)
print('KNeighborsClassifier : '+str(knn.score(x_test,y_test)*100))


labels = ["4G-supported",'Not supported']
values=d['4g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()

sns.boxplot(x="rating_cat", y="main_camera", data=d)
#sns.boxplot(x="rating_cat", y="display_size", data=d)
#ind = np.arange(5) 

ind = np.arange(10)
plt.clf()
#plt.yticks(ind,('10','20','30','40','50'))
plt.xlabel('Folds')
plt.ylabel('Accuracy')
plt.plot(score1*100,'r--',label='Linear Regression')
plt.plot(score2*100,'b--',label='Decision Tree ')
plt.plot(score3*100,'g--',label='Random Forest ')
plt.plot(score4*100,'y--',label='KNN')

plt.legend()
plt.show()