import os
import joblib
import pandas as pd
from pandas import read_pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



data_folder = "data"
dfs=[]

for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder,filename)
    if filename.endswith(".pkl"):
        df = read_pickle(file_path)
        dfs.append(df)
        print(f"file read: {filename}")


if dfs:
    combined_df = pd.concat(dfs,ignore_index=True)
else:
    print("No pickle files")
    combined_df = pd.DataFrame()

# print(combined_df.head())
# print(combined_df.tail())
# print(combined_df.info())
# #checking missing values
# print(combined_df.isnull().sum()) #NO MISSING VALUES IN DATA SET
# #distrubutions of legit transactions and fraudulent transactions
# #0 -> Normal transaction and 1 -> Fraud transactions
# print(combined_df['TX_FRAUD'].value_counts()) #THIS DATASET IS HIGHLY UNBALANCE
#Separating the data for analysis
legit = combined_df[combined_df.TX_FRAUD==0]
fraud = combined_df[combined_df.TX_FRAUD==1]
# print(legit.shape)
# print(fraud.shape)
# # statistical measure of data
# print(legit.TX_AMOUNT.describe())
# print(fraud.TX_AMOUNT.describe())
# # compare the values for both transactions
combined_df.groupby('TX_FRAUD').mean()
#UNDER SAMPLING
#built a sample data set containing similar distribution of normal and fraud transactions
#the number of fraudulent transactions ---> 14681
#the number of legit transactions ---> 1739474
# now we are going to take random 14681 transactions from legit transactions and join with 14681 fraud transaction,
# it will be a very good data set
legit_sample = legit.sample(n=14681)
# concatenating two DataFrames
new_df = pd.concat([legit_sample,fraud],axis=0) #axis=0 make sure that dataframe is added one by one below the legit_sample
#axis=0 represents row and axis=1 represent column
# print(new_df.head())
# print(new_df.tail())
#print(new_df['TX_FRAUD'].value_counts()) #now legit and fraud transactions are equal
new_df.groupby('TX_FRAUD').mean()
#SPLITTING THE DATA INTO FEATURES AND TARGETS(targets are nothing but 0 or 1)
X = new_df.drop(columns='TX_FRAUD',axis=1)  #storing all the features in X and axis =1 means storing column by column
Y = new_df['TX_FRAUD'] #storing targets (i.e 0 or 1)
# print(X)
# print(Y)
#SPLITTING THE DATA INTO TRAINING DATA AND TESTING DATA
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
#Here test_size=0.2 represent 20% data goes to testing and 80% data goes to training
# print(X.shape,X_train.shape,X_test.shape)

X_train["TX_HOUR"] = X_train["TX_DATETIME"].dt.hour
X_train["TX_DAY_OF_WEEK"] = X_train["TX_DATETIME"].dt.dayofweek
X_train["TX_DAY"] = X_train["TX_DATETIME"].dt.day

X_test["TX_HOUR"] = X_test["TX_DATETIME"].dt.hour
X_test["TX_DAY_OF_WEEK"] = X_test["TX_DATETIME"].dt.dayofweek
X_test["TX_DAY"] = X_test["TX_DATETIME"].dt.day
#Drop the original datetime column
X_train = X_train.drop(columns=["TX_DATETIME"])
X_test = X_test.drop(columns=["TX_DATETIME"])

#MODEL TRAINING
#Logistic Regression for binary classification
#model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga', random_state=42) #we are loading one instance of LogisticRegression into model variable
#model = LogisticRegression()
model = make_pipeline(StandardScaler(), LogisticRegression())
#taining the Logistic Regression model with training data

#scaler = StandardScaler()

# Fit and transform the training data
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame with original column names
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Train model
model.fit(X_train, Y_train)
#MODEL EVALUATION
#ACCURACY SCORE
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print(f'Accuracy: {training_data_accuracy}')
#accurancy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print(f'Accuracy_test: {test_data_accuracy}')

expected_features = joblib.load("expected_features.pkl")
# print("Expected Features:", expected_features)

joblib.dump(list(X_train.columns), "expected_features.pkl")
joblib.dump(model, "fraud_detection_model.pkl")
# joblib.dump(scaler, "scaler.pkl")
