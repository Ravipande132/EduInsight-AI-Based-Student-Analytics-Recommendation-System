import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("student_data.csv")
df.head()

df.isnull().sum()

# Fill missing values
df.fillna(df.mean(), inplace=True)

sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

X = df.drop("Final_Result", axis=1)
y = df["Final_Result"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

print("LR Accuracy:", accuracy_score(y_test, lr.predict(X_test)))
print("DT Accuracy:", accuracy_score(y_test, dt.predict(X_test)))
print("RF Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

def recommend(row):
    tips = []
    
    if row['Study Hours'] < 3:
        tips.append("Increase study hours")
    if row['Attendance'] < 75:
        tips.append("Improve attendance")
    if row['Sleep Hours'] < 6:
        tips.append("Get proper sleep")
        
    return tips

df['Recommendations'] = df.apply(recommend, axis=1)
df[['Final_Result','Recommendations']].head()

def recommend(row):
    tips = []
    
    if row['Study Hours'] < 3:
        tips.append("Increase study hours")
    if row['Attendance'] < 75:
        tips.append("Improve attendance")
    if row['Sleep Hours'] < 6:
        tips.append("Get proper sleep")
        
    return tips

df['Recommendations'] = df.apply(recommend, axis=1)
df[['Final_Result','Recommendations']].head()
