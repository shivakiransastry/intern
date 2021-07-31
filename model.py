import pandas as p
import pickle
import numpy as np

data=p.read_csv('odi stat.csv',header=0,encoding='unicode_escape')

dataset=['Innings','Runs','Average','Strike Rate']
ds=data[dataset]
ds.to_csv('csv1.csv',index=False) 
#print(ds)
x=ds.iloc[:,[0,1,3]].values
y=ds.iloc[:,2].values

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
reg=LinearRegression()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=44)
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)
print('efficiency of the algorithm respect to dataset :',reg.score(x_test,y_test))

pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[452,18426,86.3]]))