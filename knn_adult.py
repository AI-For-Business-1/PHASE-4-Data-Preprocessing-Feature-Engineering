
# Step 1: Import Required Libraries

# Numerical computations
import numpy as np

# Data manipulation
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('/content/adult.csv')
df.head(5)

df.info()

df.shape

df.describe()

df.isna().sum()

df.isnull().sum().sort_values()

df["education"].value_counts()

df["workclass"].value_counts()

df["occupation"].value_counts()

df["income"].value_counts()

df[df == '?'] = np.nan

for col in ['workclass', 'occupation', 'native-country']:
    df[col] = df[col].fillna(df[col].mode()[0])

df.isnull().sum()

df["occupation"].value_counts()

df.duplicated().sum()

"""Summary of variables

There are 10 numerical variables in the dataset.

All of the variables are of discrete type.

Out of all the 10 variables, the first 9 variables are feature variables and
last variable Class is the target vari
"""

df.drop_duplicates(inplace=True)

df.duplicated().sum()

X = df.drop(columns="income")
y=df["income"]

X.head()

"""# Split data into separate training and test set"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

"""# Feature Engineering"""

from sklearn import preprocessing

categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

"""# Feature Scaling"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)



"""# Model KNN"""

knn=KNeighborsClassifier(n_neighbors=5) # class wigth
knn.fit(X_train,y_train)

y_pred_train=knn.predict(X_train)
y_pred_test=knn.predict(X_test)
acc_train= accuracy_score(y_train, y_pred_train)
acc_test= accuracy_score(y_test, y_pred_test)
print("Train set accuracy: ", acc_train)
print("Test set accuracy: ", acc_test)

"""# Print the Confusion Matrix

# Train
"""

from sklearn.metrics import confusion_matrix

cm_train = confusion_matrix(y_train, y_pred_train)
print(' Confusion Matrix train \n',cm_train)

cm_df_train = pd.DataFrame(cm_train, index = ['<=50K', '>50K'], columns = ['<=50K', '>50K'])
sns.heatmap(cm_df_train, annot=True)

from sklearn.metrics import classification_report

print(classification_report(y_train, y_pred_train))

"""# Test"""

cm_test = confusion_matrix(y_test, y_pred_test)
print(' Confusion Matrix test \n',cm_test)

cm_df_test = pd.DataFrame(cm_train, index = ['<=50K', '>50K'], columns = ['<=50K', '>50K'])
sns.heatmap(cm_df_test, annot=True)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))