import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

url="https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df=pd.read_csv(url)
df=df[df["BloodPressure"]>0]
X=df[["Pregnancies","Glucose","BMI","Age","DiabetesPedigreeFunction","Insulin","Glucose","SkinThickness"]]
y=df["Outcome"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("Medical AI accuracy score:",accuracy_score(y_test,predictions))