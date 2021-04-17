# importing pandas for data frame, numpy for reshape
import pandas as pd
import numpy as np
  
# loading dataset and storing in train variable
train=pd.read_csv('station_day.csv')
  
# display top 5 data
train.head()



#cleaning the datatset
df = pd.DataFrame(train)
train=df.dropna()
train.head()



#define function that will classify AQI according to the six levels
def classify(aqi):
    if aqi<=50:
        print("Air Quality : Good\nAir Quality is satisfactory, and air pollution poses little or no risk.")
    elif aqi<=100:
        print("Air Quality : Moderate\nAir quality is acceptable. \nHowever, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.")
    elif aqi<=150:
        print("Air Quality : Unhealthy for sensitive groups\nMembers of sensitive groups may experience health effects. \nThe general public is less likely to be affected.")
    elif aqi<=200:
        print("Air Quality : Unhealthy\nSome members of the general public may experience health effects; \nmembers of sensitive groups may experience more serious health effects.")
    elif aqi<=300:
        print("Air Quality : Very Unhealthy\nHealth alert: The risk of health effects is increased for everyone.")
    else:
        print("Air Quality : Hazardous\nHealth warning of emergency conditions: everyone is more likely to be affected.")
       
      
      
# RandomForest model
# importing Randomforest
from sklearn.ensemble import RandomForestRegressor

# defining model
m1 = RandomForestRegressor()
  
# seperating class label and other attributes
train1 = train.drop(['AQI'], axis=1)
target = train['AQI']
  
# Fitting the model
m1.fit(train1, target)
  
# predicting with other values (testing the data)
# so AQI is 184
aqi1=m1.predict([[81.4, 124.5, 20.5, 0.12, 15.24, 127.09]])
print(aqi1[0])
classify(aqi1[0])



# Adaboost model
# importing module
from sklearn.ensemble import AdaBoostRegressor

# defining model
m2 = AdaBoostRegressor()
  
# Fitting the model
m2.fit(train1, target)
  
# predicting the model with other values (testing the data)
# so AQI is 184
aqi2=m2.predict([[81.4, 124.5, 20.5, 0.12, 15.24, 127.09]])
print(aqi2[0])
classify(aqi2[0])



#taking values from user and predicted the AQI
val=[None]*6
val[0]=float(input("Enter PM2.5 value: "))
val[1]=float(input("Enter PM10 value: "))
val[2]=float(input("Enter NO2 value: "))
val[3]=float(input("Enter CO value: "))
val[4]=float(input("Enter SO2 value: "))
val[5]=float(input("Enter O3 value: "))
print(val)



#predicting AQI using RandomForest model
predict1=m1.predict(np.array(val).reshape(1,-1))
print(predict1[0])
classify(predict1[0])


#predicting AQI using AdaBoost model
predict2=m2.predict(np.array(val).reshape(1,-1))  
print(predict2[0])
classify(predict2[0])

