import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pickle

#load data

data=pd.read_csv("data/House_Rent_Dataset.csv")

# Drop non-numeric columns that won't be used for prediction
data = data.drop(['Posted On', 'Floor', 'Area Locality', 'Point of Contact'], axis=1)

#encode categorical Variables

data=pd.get_dummies(data,columns=['Area Type','City','Furnishing Status','Tenant Preferred'],drop_first=True)

#Feature Engineering
data['Rent_per_sqft']=data['Rent']/data['Size']
data['Size_per_BHK']=data['Size']/data['BHK']

data['Bathroomm_to_BHK_ratio']=data['Bathroom']/data['BHK']

#feature and target

X=data.drop(['Rent','Rent_per_sqft'],axis=1)
y=data['Rent']

#train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#scale features

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


#Trying Multiple models to findout best models

models={

    'LinearRegression':LinearRegression(),
    'Random Forest':RandomForestRegressor(),
    'GradientBoostingRegressor':GradientBoostingRegressor()

}

results={}

for name, model in models.items():

    print(f"\n{'='*50}")
    print(f"Traning{name}...")
    print('='*50)

    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)



    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)


    results[name]={'RSME':rmse,"MAE":mae,'R-Square':r2}

    print(f"RMSE: Rs.{rmse:,.2f}")
    print(f"MAE: Rs.{mae:,.2f}")
    print(f"R-Square Score:{r2:,.4f}")
    print(f"Mean Rent: Rs.{y_test.mean():,.2f}")
    print(f"Prediction Accuracy(MAE/Mean):{(mae/y_test.mean())*100:.2f}%")

#select the Best Model based on R2

best_model_name=max(results,key=lambda x: results[x]['R-Square'])
best_model=models[best_model_name]

print(f"\n{'='*50}")

print( f"Beast model:{best_model_name}")
print(f"r2 score:{results[best_model_name]['R-Square']:.4f}")

print('='*50)

#save the best model and scalar
with open('saved_model/rent_model.pkl','wb') as f:
    pickle.dump(best_model,f)

with open('saved_model/scalara.pkl','wb') as f:
    pickle.dump(scaler,f)

print("\n Model and Scalar saved Sucessfully..")