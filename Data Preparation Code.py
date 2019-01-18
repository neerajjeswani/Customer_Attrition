#Importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation

#Load Data
t = pd.read_csv("/Users/neerajjeswani/Desktop/Telco/TCC.csv")
t.head()

#Overall Summary
print ("Rows     : " ,t.shape[0])
print ("Columns  : " ,t.shape[1])
print ("Features : \n" ,t.columns.tolist())
print ("Missing values :  ", t.isnull().sum().values.sum())
print ("Unique values in each Feature :  \n",t.nunique())

print(type(t["TotalCharges"][1]))

#Replacing Blanks
t['TotalCharges'] = t["TotalCharges"].replace(" ",np.nan)
t = t[t["TotalCharges"].notnull()]
t = t.reset_index()[t.columns]

#Converting data type
t["TotalCharges"] = t["TotalCharges"].astype(float)

print(type(t["TotalCharges"][1]))

#Finding at Unique Values
for col in t:
    print(col, t[col].unique())

#Replacing 'No Internet Service'
to_replace = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies']

for col in to_replace : 
    t[col]  = t[col].replace({'No internet service' : 'No'})

#Changing Senior Citizen variable to 'Yes/No'
t["SeniorCitizen"] = t["SeniorCitizen"].replace({1:"Yes",0:"No"})

#Bucketting Tenure
def buckets(df) :
    
    if df["tenure"] <= 12 :
        return "0-12"
    elif (df["tenure"] > 12) & (df["tenure"] <= 24 ):
        return "12-24"
    elif (df["tenure"] > 24) & (df["tenure"] <= 48) :
        return "24-48"
    elif (df["tenure"] > 48) & (df["tenure"] <= 60) :
        return "48-60"
    elif df["tenure"] > 60 :
        return ">60"
    
t["tenure_bucket"] = t.apply(lambda t:buckets(t),axis = 1)
t.drop("tenure", axis=1, inplace=True)

#Differentiating Columns into Categorical & Numerical
Id_col     = ['customerID']
target_col = ["Churn"]
cat_cols   = t.nunique()[t.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in t.columns if x not in cat_cols + Id_col]

df1 = t[cat_cols]
df2 = t[num_cols]
df=pd.merge(left=df1, right=df2, left_index=True, right_index=True)

df["Churn"] = df["Churn"].replace({"Yes":1,"No":0})

#Converting categorical into numerical
cols = df.columns

labels = []

for i in range(0,17):
    train = df[cols[i]].unique()
    labels.append(list(set(train))) 

cats = []

for i in range(0, 17):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(df.iloc[:,i])
    feature = feature.reshape(df.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = np.column_stack(cats)

df = np.concatenate((encoded_cats,df.iloc[:,17:].values),axis=1)

#Creating X & Y dataframes
X = df[:,0:44]
Y = df[:,44]

#Splitting into Train and Test Data
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=.2)

#Saving Data
np.savetxt("/Users/neerajjeswani/Desktop/Telco/X_train.csv", X_train, delimiter=",")
np.savetxt("/Users/neerajjeswani/Desktop/Telco/X_val.csv", X_val, delimiter=",")
np.savetxt("/Users/neerajjeswani/Desktop/Telco/Y_train.csv", Y_train, delimiter=",")
np.savetxt("/Users/neerajjeswani/Desktop/Telco/Y_Val.csv", Y_val, delimiter=",")