import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Gender_Number"]=df["Sex"].map({"female":1,"male":0})

X=df[["Pclass","Age","Gender_Number"]]
y=df["Survived"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

model=DecisionTreeClassifier(max_depth=3)
model.fit(X_train,y_train)

predictions=model.predict(X_test)
final_grade=accuracy_score(y_test,predictions)
print("AI final score:",final_grade)
# 7. Print the Confusion Matrix
matrix = confusion_matrix(y_test, predictions)

print("\n--- CONFUSION MATRIX ---")
print("                 [Predicted Dead]  [Predicted Survived]")
print("[Actually Dead]       ", matrix[0][0], "               ", matrix[0][1])
print("[Actually Survived]   ", matrix[1][0], "               ", matrix[1][1])
