# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd 
from scipy import stats 
import numpy as np
df=pd.read_csv("bmi.csv") 
df.head()
```

<img width="362" height="197" alt="image" src="https://github.com/user-attachments/assets/af83155a-e2f6-4026-b196-cbb295017fb3" />


```
df_null_sum=df.isnull().sum() 
df_null_sum
```

<img width="223" height="106" alt="image" src="https://github.com/user-attachments/assets/94df3f63-aa6c-48df-8e18-2d2698cdc5cd" />

```
df.dropna()
```

<img width="367" height="360" alt="image" src="https://github.com/user-attachments/assets/d9323cab-2af1-4829-891b-1a22a9c6f76f" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0) 
max_vals
```

<img width="200" height="71" alt="image" src="https://github.com/user-attachments/assets/75c38991-bb11-41e9-9b10-90a20d711d72" />

```
from sklearn.preprocessing import StandardScaler 
df1=pd.read_csv("bmi.csv") 
df1.head()
```

<img width="352" height="191" alt="image" src="https://github.com/user-attachments/assets/508920a3-3d8a-4e47-996c-1c9f4c3df871" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']]) 
df1.head(10)
```

<img width="412" height="322" alt="image" src="https://github.com/user-attachments/assets/04863273-8235-4db6-9c66-8f69992a1727" />

```
from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler() 
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="395" height="314" alt="image" src="https://github.com/user-attachments/assets/a38c7429-867f-40fa-b5df-4e04d618295a" />

```
from sklearn.preprocessing import MaxAbsScaler 
scaler = MaxAbsScaler() 
df3=pd.read_csv("bmi.csv") 
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']]) 
df
```

<img width="392" height="363" alt="image" src="https://github.com/user-attachments/assets/5691e89b-dd0b-40a1-a9a2-4c241c20b976" />

```
from sklearn.preprocessing import RobustScaler 
scaler = RobustScaler() 
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']]) 
df3.head()
```

<img width="399" height="165" alt="image" src="https://github.com/user-attachments/assets/6380148d-cf9e-47fb-92a4-fadf3b20c1ef" />

```
df=pd.read_csv("income(1) (1).csv") 
df.info()
```

<img width="419" height="356" alt="image" src="https://github.com/user-attachments/assets/d9a664a4-b82e-4192-9993-af7618b31004" />

```
df_null_sum=df.isnull().sum() 
df_null_sum
```

<img width="321" height="250" alt="image" src="https://github.com/user-attachments/assets/9a0f4e60-47d3-44d8-b725-4901e8402766" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

<img width="704" height="498" alt="image" src="https://github.com/user-attachments/assets/f10bf1b3-ece7-4a01-959c-1efd8856d5e3" />

```
df[categorical_columns] = df[categorical_columns].astype('category') 
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes) 
df[categorical_columns]
```

<img width="722" height="380" alt="image" src="https://github.com/user-attachments/assets/58b5be13-6e2b-4921-87f8-a203f95ab0fc" />

```
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
```

<img width="386" height="102" alt="image" src="https://github.com/user-attachments/assets/c4185ef4-0425-4d08-bbdd-eee8d1a48ca3" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("income(1) (1).csv") 
df.info()
```

<img width="379" height="346" alt="image" src="https://github.com/user-attachments/assets/d4ec62bf-96a2-4e39-8ca5-06ceb642fe6c" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

<img width="707" height="490" alt="image" src="https://github.com/user-attachments/assets/b768b931-3f01-40a9-8647-403d05e6719c" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="677" height="368" alt="image" src="https://github.com/user-attachments/assets/ef7c1a08-1b4b-4068-a6b5-ee0eaf3117db" />

```
X = df.drop(columns=['SalStat']) 
y = df['SalStat'] 
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2) 
X_chi2 = selector_chi2.fit_transform(X, y) 
selected_features_chi2 = X.columns[selector_chi2.get_support()] 
print("Selected features using chi-square test:") 
print(selected_features_chi2) 
```

<img width="611" height="94" alt="image" src="https://github.com/user-attachments/assets/92bdc874-47ad-43dd-b9a5-4a293eb68b73" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss', 'hoursperweek'] 
X = df[selected_features] 
y = df['SalStat'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
```

<img width="356" height="94" alt="image" src="https://github.com/user-attachments/assets/1d9e1bba-fa30-4fc0-9b66-2cc0d9d85a89" />

```
y_pred = rf.predict(X_test) 
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Model accuracy using selected features: {accuracy}")
```

<img width="500" height="36" alt="image" src="https://github.com/user-attachments/assets/017f6fff-d5bb-4189-a974-5e83e00612bb" />

```
!pip install skfeature-chappers
```

<img width="634" height="386" alt="image" src="https://github.com/user-attachments/assets/657889b0-31dc-4e75-b768-66996b8547ed" />

```
import numpy as np 
import pandas as pd 
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
categorical_columns = [ 'JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry' ]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

<img width="691" height="361" alt="image" src="https://github.com/user-attachments/assets/8d659408-0ecc-4627-876d-74fba80d478c" />

```
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
k_anova = 5 
selector_anova = SelectKBest(score_func=f_classif,k=k_anova) 
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```

<img width="623" height="72" alt="image" src="https://github.com/user-attachments/assets/3b39c878-02dc-4352-8c92-cbbb0f4e8ce6" />


```
X = df.drop(columns=['SalStat']) 
y = df['SalStat'] 
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select) 
rfe.fit(X, y)
```

<img width="826" height="847" alt="image" src="https://github.com/user-attachments/assets/338e7e40-4da5-4398-8894-0ca786d23aa8" />

# RESULT:

The given dataset was successfully read, and Feature Scaling and Feature Selection were performed. The selected and scaled features were saved successfully into a new data file.
