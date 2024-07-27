import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def checkOutcome(rawvalue):
    checkHospitalized = ["discharged", "hospitalized","critical condition","discharge"]
    checkNonHosp = ["alive", "treatment", "stable", "recovering", "quarantine"]
    checkDeceased = ["dead", "death", "deceased", "died"]
    if any(x in rawvalue.lower() for x in checkHospitalized):
        return "hospitalized"
    elif any(x in rawvalue.lower() for x in checkNonHosp):
        return "nonhospitalized"
    elif any(x in rawvalue.lower() for x in checkDeceased):
        return "deceased"
    elif "recovered" in rawvalue.lower():
        return "recovered"
    return None

def predictProvinceKnn(cases_train):
    """We can predict/impute province accurately based on the latitude and longitude and country given, we can use Knn Algorithm to predict the missing values, it also makes sense,
    nearer the point to the cluster of longitude and latitude the high chances that it is that cluster province."""
    #Seperating missing values first to train and test model
    tmp = cases_train.copy()
    pd.options.mode.chained_assignment = None
    #using labelencoder to utilize categorical data (country) in KNN algo.
    le = LabelEncoder()
    tmp['country_encode'] = le.fit_transform(tmp['country'])
    X = tmp[tmp['province'].notnull()]
    y = X['province']
    finalImpute = tmp[~tmp.index.isin(X.index)]

    #creating training data for Knn
    X_train, X_test, y_train, y_test = train_test_split(X[['country_encode','latitude','longitude']],y)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)

    #our model predictions look very promising
    print("Knn score: ",knn.score(X_test,y_test))

    #Predicting provinces with our trained model
    finalImpute['province'] = knn.predict(finalImpute[['country_encode','latitude','longitude']])
    t = finalImpute.append(X)
    t.sort_index(inplace = True)
    t.drop(['country_encode'],axis = 1,inplace = True)
    return t

def cleanMessy1_1(cases_train):
    cases_train['outcome_group'] = cases_train.outcome.apply(checkOutcome)
    cases_train.drop("outcome",axis = 1,inplace = True)
    cases_train.to_csv('data/cases_2021_train.csv', index=False)
    print("1.1 Number of cases:")
    print(cases_train.groupby(by='outcome_group').size())
    return cases_train
    
def cleanAge(agestr):    
    tmp = agestr.split("-")
    if '' in tmp:
        tmp.remove('')
    for i in range(0, len(tmp)):
        tmp[i] = float(tmp[i])
    ageround = 1.0*sum(tmp)/len(tmp)
    return round(ageround)

def cleanandImpute1_4(cases,outpt):
    cases.dropna(subset = ['age'],inplace =True )
    cases['age'] = cases['age'].apply(cleanAge)
    #we cannot impute values for them plus we cannot drop them as we will lose other valuable attribute data  with them
    cases['source'].fillna('unknown',inplace = True)
    cases['additional_information'].fillna('noInfo',inplace = True)
    cases['sex'].fillna('unknown',inplace = True)
    #by manual looking and examining the eda.py missing values it seems Taiwan was the only row with missing country, in both csvs.
    cases['country'].fillna('Taiwan',inplace = True)
    cases['date_confirmation'].fillna(cases['date_confirmation'].value_counts().index[0],inplace = True)
    cases = predictProvinceKnn(cases)
    cases.reset_index()
    cases.to_csv(outpt,index = False)
    return cases

def cleanandImputeLoc1_4(location,outpt):
    location = location[~(location.Long_.isnull() & location.Lat.isnull())]
    location['Province_State'].fillna('unknown',inplace=True)
    location = location[location['Case_Fatality_Ratio'].notna()]
    location.to_csv(outpt,index = False)
    return location

def join1_6(location,cases_train,cases_test):
    location_processed = location.replace({'Country_Region': { 'US' : 'United States', 'Korea, South' : 'South Korea','Taiwan*': 'Taiwan'}})
    location_processed = location_processed.groupby(['Province_State', 'Country_Region']).\
        agg({'Confirmed':'sum', 'Deaths':'sum', 'Recovered':'sum', 'Active':'sum', 'Incident_Rate':'mean', 'Case_Fatality_Ratio':'mean'}).reset_index()
    location_processed = location_processed.rename(columns={'Country_Region': 'country', 'Province_State': 'province'})
    print(len(train))
    train_processed = pd.merge(cases_train, location_processed, on=["province", "country"],how='inner')
    test_processed = pd.merge(cases_test, location_processed, on=["province", "country"],how='inner')
    test_processed.to_csv('results/cases_2021_test_processed.csv',index = False)
    train_processed.to_csv('results/cases_2021_train_processed.csv',index = False)
    location_processed.to_csv('results/location_2021_processed.csv',index = False)
    print("Rows in joined cases_train:", len(train_processed))
    print("Rows in joined cases_test:", len(test_processed))
    
#https://medium.com/clarusway/z-score-and-how-its-used-to-determine-an-outlier-642110f3b482#:~:text=If%20you%20know%20the%20mean,Z%2DScore%20to%20determine%20outliers.
def zscoreOutlierDetect(location,test,train):
    listQuant = ['age','latitude','longitude','Lat', 'Long_','Confirmed', 'Deaths', 'Recovered', 'Active','Incident_Rate', 'Case_Fatality_Ratio']
    for col in listQuant:
        col_zscore = col + "_zscore"
        if col in ['age','latitude','longitude']:
            train[col_zscore] = (train[col] - train[col].mean())/train[col].std(ddof=0)
            train["outlier"] = (abs(train[col_zscore])>3).astype(int)
            print("Outliers in Train",col,":",len(train[train.outlier == 1]))
            if str(col) != 'age':
                train = train[train.outlier == 0]
            train.drop(col_zscore,inplace = True,axis = 1)

            test[col_zscore] = (test[col] - test[col].mean())/test[col].std(ddof=0)
            test["outlier"] = (abs(test[col_zscore])>3).astype(int)
            print("Outliers in Test",col,":",len(test[test.outlier == 1]))
            if str(col) != 'age':
                test = test[test.outlier == 0]
            test.drop(col_zscore,inplace = True,axis = 1)
        else:
            location[col_zscore] = (location[col] - location[col].mean())/location[col].std(ddof=0)
            location["outlier"] = (abs(location[col_zscore])>3).astype(int)
            print("Outliers in Location",col,":",len(location[location.outlier == 1]))
            location = location[location.outlier == 0]
            location.drop(col_zscore,inplace = True,axis = 1)
    location.drop('outlier',inplace = True,axis = 1)
    train.drop('outlier',inplace = True,axis = 1)
    test.drop('outlier',inplace = True,axis = 1)
    return location, test, train
    
def main():
    cases_train = pd.read_csv('data/cases_2021_train.csv')
    cases_test = pd.read_csv('data/cases_2021_test.csv')
    cases_train.drop_duplicates(inplace = True)
    cases_test.drop_duplicates(inplace = True)
    location = pd.read_csv('data/location_2021.csv')
    #1.1
    cases_train = cleanMessy1_1(cases_train)
#     #1.2
#     #Feature Extraction
    
#     #1.4
#     print("1.4")
#     cases_train = cleanandImpute1_4(cases_train,'data/cases_2021_train.csv')
#     cases_test = cleanandImpute1_4(cases_test,'data/cases_2021_test.csv')
#     location = cleanandImputeLoc1_4(location,'data/location_2021.csv')
    
    # #1.5
    # location, cases_test, cases_train = zscoreOutlierDetect(location,cases_test,cases_train)
    # print("Rows in joined cases_train:", len(cases_train))
    # print("Rows in joined cases_test:", len(cases_test))
    # #1.6
    # print("1.6")
    # join1_6(location,cases_train,cases_test)
    
if __name__ == '__main__':
    main();