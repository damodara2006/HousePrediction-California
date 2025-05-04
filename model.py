import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle


def dataload():
    df = pd.read_csv('housing.csv')
    df.dropna( inplace=True)
    df['total_rooms'] = np.log(df['total_rooms'] + 1)
    df['total_bedrooms'] = np.log(df['total_bedrooms'] + 1)
    df['population'] = np.log(df['population'] + 1)
    df['households'] = np.log(df['households'] + 1)
    df = df.join(pd.get_dummies(df["ocean_proximity"])).drop('ocean_proximity',axis=1)
    return df
    
    
def model(df):
    # print(df)
    x = df.drop('median_house_value',axis=1) 
    y = df['median_house_value']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # print(y_train)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

   
    # print(res)
    return model
    
def predict(Model , data):
    data = pd.DataFrame([data])
    return Model.predict(data)
       
       
    
def main():
    df = dataload()
    Model = model(df)
    
    
    with open("model.pkl","wb") as f:
        pickle.dump(Model, f)
    
    
if __name__ == '__main__':
        main()