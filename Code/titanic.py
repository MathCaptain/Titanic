"""
THE PURPOSE OF THIS SCRIPT IS TO ALLOW ME TO PRACTICE THE STEPS INVOLVED IN 
WORKING WITH A KAGGLE DATASET, FROM DATA PREPROCESSING THROUGH TO PREDICTION
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

""" Getting the input Data Sets """

# Setting the Working directory
import os
path = "C:/Users/ssaawi/OneDrive - SAS/Documents/Python/Kaggle/Titanic/Data"
os.chdir(path)
print(os.listdir())

# Importing the Training and Test Sets

data_train_file = "train.csv"
data_test_file = "test.csv"

df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)

# Investigating the data
df_train.head()
df_train.describe()

# List of variable names as console wasn't printing all the columns
print(df_train.columns)


""" Detecting Outliers """

def outlier_detection(df_train,variables,cuttoff):
    """
    df_train - The training dataset
    variables - what variables you want to use in checking for outliers
    cuttoff - How many variables need be identified as an outlier for that 
    observation to be removed
    """
    outlier_indicator = pd.DataFrame(np.zeros(df_train[variables].shape),columns=variables)
    
    for variable in variables:
        Q1 = np.percentile(df_train[variable], 25)
        Q3 = np.percentile(df_train[variable], 75)
        IQR = Q3 - Q1
        Outlier_range = 1.5*IQR

        outlier_indicator[variable] = np.logical_or(df_train[variable] < Q1-Outlier_range,df_train[variable] > Q3+Outlier_range)

    outlier_count = outlier_indicator.sum(axis=1)
    
    outliers = pd.DataFrame(outlier_count[outlier_count >= cutoff])
    
    return outliers

variables = ["Age","SibSp","Parch","Fare"]
cutoff = 3
outliers = outlier_detection(df_train,variables,cutoff)

# Looking at our outliers
df_train.loc[outliers.index,:]

# Saving as a different dataframes as I don't want to lose data
df_train2 = df_train.loc[~df_train.index.isin(outliers.index),:].reset_index(drop=True)
df_train2

# Checking Amount of Rows in new dataset
df_train2.info() #we can infer missing values from this also
df_train2.isnull().sum() #count of missing values for each column

df_test.info() #we can infer missing values from this also
df_test.isnull().sum() #count of missing values for each column
# Note missing value of "Fare" in TEST set. We should impute based on inferences from TRAINING set


""" Dealing with Missing Values """

# Investigate correlation between Age and other variables
corr_train = df_train2.loc[:,['Age','Fare','Parch','SibSp']].corr()

#Correlation Matrix for Numeric Variables
plt.subplots(figsize=(12,10))
sns.heatmap(corr_train,annot=True,vmin=-0.4,vmax=0.4,cmap='GnBu_r')

# Box Plot for categorical variables against AGE
sns.boxplot(data=df_train2,y='Age',x='Sex')
sns.boxplot(data=df_train2,y='Age',x='Pclass')
sns.boxplot(data=df_train2,y='Age',x='Embarked')
sns.boxplot(data=df_train2,y='Age',x='Parch')
sns.boxplot(data=df_train2,y='Age',x='SibSp')

# Both Pclass and SibSp Seem to be correlated with Age
# Checking to see of Age differs for SibSp grouped by Pclass
plt.subplots(figsize=(15,15))
sns.boxplot(data=df_train2,y='Age',x='SibSp',hue='Pclass')

# Box Plot for categorical variables against FARE
sns.boxplot(data=df_train2,y='Fare',x='Sex',showfliers=False)
sns.boxplot(data=df_train2,y='Fare',x='Pclass',showfliers=False)
sns.boxplot(data=df_train2,y='Fare',x='Embarked',showfliers=False)
sns.boxplot(data=df_train2,y='Fare',x='Parch',showfliers=False)
sns.boxplot(data=df_train2,y='Fare',x='SibSp',showfliers=False)

""" TRAINING IMPUTATION """
# Getting the Index of the Missing values for Age
Missing_Age_Index = df_train2[df_train2.Age.isnull()].index

# Finding Median for combinations of Pclass and SibSp
# Just grouping by Sibsp only when Sibsp >= 3 
for i in Missing_Age_Index:
    if df_train2.loc[i,"SibSp"] >= 3:
        df_train2.loc[i,"Age"] = df_train2.Age[(df_train2.loc[:,"SibSp"] == df_train2.loc[i,"SibSp"])].median()
    else:
        df_train2.loc[i,"Age"] = df_train2.Age[((df_train2.loc[:,"SibSp"] == df_train2.loc[i,"SibSp"]) &
            (df_train2.loc[:,"Pclass"] == df_train2.loc[i,"Pclass"]))].median()
 
# Checking Age no longer has any missing values
df_train2.Age.isnull().sum()

# Only 2 missing values of Embarked so will just impute the mode
Missing_Embarked_Index = df_train2[df_train2.Embarked.isnull()].index

df_train2.Embarked.fillna(df_train2.Embarked.mode()[0],inplace=True)
df_train2.loc[Missing_Embarked_Index,'Embarked'] #checking missings were imputed
df_train2.Embarked.isnull().sum()

""" TEST IMPUTATION """
# Getting the Index of the Missing values for Age
Missing_Age_Index_Test = df_test[df_test.Age.isnull()].index

# Finding Median (OF TRAINING SET) for combinations of Pclass and SibSp
# Just grouping by Sibsp only when Sibsp >= 3 
for i in Missing_Age_Index_Test:
    if df_test.loc[i,"SibSp"] >= 3:
        df_test.loc[i,"Age"] = df_train2.Age[(df_train2.loc[:,"SibSp"] == df_test.loc[i,"SibSp"])].median()
    else:
        df_test.loc[i,"Age"] = df_train2.Age[((df_train2.loc[:,"SibSp"] == df_test.loc[i,"SibSp"]) &
            (df_train2.loc[:,"Pclass"] == df_test.loc[i,"Pclass"]))].median()
 
# Checking Age no longer has any missing values
df_test.Age.isnull().sum()

# Checking the observation that still has missing Age
df_test[df_test.Age.isnull()]
# It is because SibSp = 8 and SibSp only goes up to 5 in df_train2


# Creating new loop to take care of values of SibSp > 5 (max value of Pcalss in df_train2)
# (Incorporate into previous loop if new test data is brought in)
Missing_Age_Index_Test2 = df_test[df_test.Age.isnull()].index
for i in Missing_Age_Index_Test2:
    if df_test.loc[i,"SibSp"] >= df_train2.SibSp.max(): #generalize code so 5 isnt hardcoded
        df_test.loc[i,"Age"] = df_train2.Age[df_train2.loc[:,"SibSp"] == df_train2.SibSp.max()].median()
    else: #For the case where Pclass is missing
        df_test.loc[i,"Age"] = df_train2.Age.median()

# Checking Age no longer has any missing values
df_test.Age.isnull().sum()


# From Box Plots have decided to impute Fare based off Sex, Pclass and Embarked
Missing_Fare_Index_Test = df_test[df_test.Fare.isnull()].index

for i in Missing_Fare_Index_Test:
    Condition = ((df_train2.loc[:,"Sex"] == df_test.loc[i,"Sex"]) &
            (df_train2.loc[:,"Pclass"] == df_test.loc[i,"Pclass"]) &
            (df_train2.loc[:,"Embarked"] == df_test.loc[i,"Embarked"]))
    if Condition.sum() > 0:
        df_test.loc[i,"Fare"] = df_train2.Fare[Condition].median()
    else: 
        #For case where combination of Sex, Pclass and Embarked from Test set doesnt exist in Training set
        #Using Pclass as clear and intuitive realation in box plot
        df_test.loc[i,"Fare"] = df_train2.Fare[(df_train2.loc[:,"Pclass"] == df_test.loc[i,"Pclass"])].median()

 
# Checking Age no longer has any missing values
df_test.Fare.isnull().sum()


""" Standardizing Features """
# I have decided to standardize the numeric (continuous) features

# Get column names first
names = ["Age","Fare"] #removed Age as it is now categorical

# Create the Scaler object
scaler = preprocessing.StandardScaler()

""" Training """
# Fit your training data on the scaler object
scaled_df_train = scaler.fit_transform(df_train2.loc[:,names])
df_train2.loc[:,names] = pd.DataFrame(scaled_df_train, columns=names)

""" Test """
# Fit your test data on the scaler object
scaled_df_test = scaler.fit_transform(df_test.loc[:,names])
df_test.loc[:,names] = pd.DataFrame(scaled_df_test, columns=names)

        
""" Feature Engineering """

df_full = [df_train2, df_test]

""" Has_Cabin """
# First will deal with the last variable with missing values: Cabin  
# Create indicator variable, 1: Has Cabin, 0: No Cabin

for df in df_full:
    print(df.loc[0:20,'Cabin'])
    for i in df.Cabin.isnull().index:
        if (df.Cabin.isnull()[i] == 1):
            df.loc[i,'Has_Cabin'] = 0
        else:
            df.loc[i,'Has_Cabin'] = 1
    
    print(df.loc[0:20,['Cabin','Has_Cabin']])
    
    
""" Title """
for df in df_full:
    # Extracting Title from Name
    for i in df.index:
        df.loc[i,'Title'] = re.split(' ([A-Za-z]+)\.', df.loc[i,'Name'])[1]
    
    print(df.head())
    print(df.Title.unique())
    
    # Checking how many values exist for each categorical value of Title
    print(df["Title"].value_counts())
    
    # Replacing rare titles with a "Uncommon" value
    for i in df.index:
        if df.loc[i,'Title'] not in ['Mr','Miss','Mrs','Master']:
            df.loc[i,'Title'] = 'Uncommon'
    
    # Checking how many values exist for each categorical value of Title
    print(df["Title"].value_counts())
    
""" Family Size """
# Combining Sibsp and Parch varibles. Allow us to see if person was alone
for df in df_full:
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df['Alone'] = df['Family_Size'].map(lambda s: 1 if s == 1 else 0)

""" Ticket Category """
# Extracting prefix to Ticket Number
for df in df_full:
    Ticket = []
    for tick in df.Ticket:
        if tick.isdigit() == False:
            Ticket.append(tick.replace(".","").replace("/","").split(' ')[0])
        else:
            Ticket.append('No Ticket Category')
    df['Ticket_Category'] = Ticket

    # Only including common categories so same amount of encoded variables in training and test sets
    values = df_train2.Ticket_Category.value_counts()
    for i in df.index:
        if df.loc[i,'Ticket_Category'] not in values.loc[values >= 5].index:
            df.loc[i,'Ticket_Category'] = 'Uncommon Ticket'


""" Age Group """
"""for df in df_full:
    # Grouping Age into buckets
    df.loc[ df['Age'] <= 5, 'Age'] 					    = 0
    df.loc[(df['Age'] > 5) & (df['Age'] <= 15), 'Age']  = 1
    df.loc[(df['Age'] > 15) & (df['Age'] <= 25), 'Age'] = 2
    df.loc[(df['Age'] > 25) & (df['Age'] <= 40), 'Age'] = 3
    df.loc[(df['Age'] > 40) & (df['Age'] <= 65), 'Age'] = 4
    df.loc[ df['Age'] > 65, 'Age']                      = 5  """

""" Variable Encoding """

# Encoding Pclass, Sex, Embarked, Title
df_train2 = pd.get_dummies(
        df_train2, 
        columns = ["Sex","Embarked","Title","Ticket_Category"],
        drop_first = True)

df_test = pd.get_dummies(
        df_test, 
        columns = ["Sex","Embarked","Title","Ticket_Category"],
        drop_first = True)
   
# Dropping unwanted variables
df_train2.drop(
        ['Name','Ticket','Cabin'], 
        axis=1,
        inplace = True)

df_test.drop(
        ['Name','Ticket','Cabin'], 
        axis=1,
        inplace = True)


""" MODEL FITTING """

# Spliting the training data set into predictors and response
X_Train = df_train2.drop(["Survived","PassengerId"], axis=1)
Y_Train = df_train2.loc[:,"Survived"]

# 5 fold cross validation
kfold = StratifiedKFold(n_splits=10)

# Setting random seed
random_state = 1234

""" Gradient Boosting """
# Setting which Classifier we want
classifier_GB = GradientBoostingClassifier(random_state=random_state)

# Estimate of the out-of-sample accuracy (using default hyperparameters)
scores = cross_val_score(classifier_GB, 
                X_Train, 
                y = Y_Train, 
                scoring = "accuracy", 
                cv = kfold)
scores.mean()

# Setting values we want to test the hyperparameters over
# Use RandomizedSearchCV when computation is taking to long with GridSearchCV
# Plus tune for important params first
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,250,500],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [3, 6],
              }

gsGBC = GridSearchCV(
        classifier_GB,
        param_grid = gb_param_grid, 
        cv=kfold, 
        scoring="accuracy", 
        n_jobs= 4, 
        verbose = 1)

gsGBC.fit(X_Train,Y_Train)

# The Best Model
GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_

# Looking at the parameters used in the model to generate this best score
gsGBC.best_estimator_

""" Neural Network """
# Setting which Classifier we want
classifier_NN = MLPClassifier(random_state=random_state)

# Estimate of the out-of-sample accuracy (using default hyperparameters)
scores = cross_val_score(classifier_NN, 
                X_Train, 
                y = Y_Train, 
                scoring = "accuracy", 
                cv = kfold)
scores.mean()

# Setting values we want to test the hyperparameters over
nn_param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

gsNN = GridSearchCV(
        classifier_NN,
        param_grid = nn_param_grid, 
        cv=kfold, 
        scoring="accuracy", 
        n_jobs= 4, 
        verbose = 1)

gsNN.fit(X_Train,Y_Train)

# The Best Model
NN_best = gsNN.best_estimator_

# Best score
gsNN.best_score_

# Looking at the parameters used in the model to generate this best score
gsNN.best_estimator_

""" Logistic Regression """
# Setting which Classifier we want
classifier_LR = LogisticRegression(random_state=random_state)

# Estimate of the out-of-sample accuracy (using default hyperparameters)
scores = cross_val_score(classifier_LR, 
                X_Train, 
                y = Y_Train, 
                scoring = "accuracy", 
                cv = kfold)
scores.mean()

# Setting values we want to test the hyperparameters over
lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

gsLR = GridSearchCV(
        classifier_LR,
        param_grid = lr_param_grid, 
        cv=kfold, 
        scoring="accuracy", 
        n_jobs= 4, 
        verbose = 1)

gsLR.fit(X_Train,Y_Train)

# The Best Model
LR_best = gsLR.best_estimator_

# Best score
gsLR.best_score_

# Looking at the parameters used in the model to generate this best score
gsLR.best_estimator_

""" Support Vector Machine """
# Setting which Classifier we want
classifier_SVMC = SVC(random_state=random_state)

# Estimate of the out-of-sample accuracy (using default hyperparameters)
scores = cross_val_score(classifier_SVMC, 
                X_Train, 
                y = Y_Train, 
                scoring = "accuracy", 
                cv = kfold)
scores.mean()

# Setting values we want to test the hyperparameters over
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000]}

gsSVMC = GridSearchCV(
        classifier_SVMC,
        param_grid = svc_param_grid, 
        cv=kfold, 
        scoring="accuracy", 
        n_jobs= 4, 
        verbose = 1)

gsSVMC.fit(X_Train,Y_Train)

# The best Model
SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_

# Looking at the parameters used in the model to generate this best score
gsSVMC.best_estimator_


""" Linear Discriminant Analysis """
# Setting which Classifier we want
classifier_LDA = LinearDiscriminantAnalysis()

# Estimate of the out-of-sample accuracy (using default hyperparameters)
scores = cross_val_score(classifier_LDA, 
                X_Train, 
                y = Y_Train, 
                scoring = "accuracy", 
                cv = kfold)
scores.mean()

# Setting values we want to test the hyperparameters over
# Default solver is svd but this can't be used with shrinkage
lda_param_grid = {'solver': ['lsqr','eigen'], 
                  'shrinkage': ['auto',0, 0.01, 0.1, 0.5, 0.9, 1]}

gsLDA = GridSearchCV(
        classifier_LDA,
        param_grid = lda_param_grid, 
        cv=kfold, 
        scoring="accuracy", 
        n_jobs= 4, 
        verbose = 1)

gsLDA.fit(X_Train,Y_Train)

# The best Model
LDA_best = gsLDA.best_estimator_

# Best score
gsLDA.best_score_
# Note slightly better performance then default values

# Looking at the parameters used in the model to generate this best score
gsLDA.best_estimator_



""" Learning Curves """
# Plotting Learning Curves using function from Scikit-Learn Documentation
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Gradient Boosting
plot_learning_curve(
        gsGBC.best_estimator_,
        "GradientBoosting learning curves",
        X_Train,
        Y_Train,
        cv=kfold)

# Neural Network
plot_learning_curve(
        gsNN.best_estimator_,
        "Neural Network learning curves",
        X_Train,
        Y_Train,
        cv=kfold)

# Logistic Regression
plot_learning_curve(
        gsLR.best_estimator_,
        "Logistic Regression learning curves",
        X_Train,
        Y_Train,
        cv=kfold)

# Support Vector Machine
plot_learning_curve(
        gsSVMC.best_estimator_,
        "Support Vector Machine learning curves",
        X_Train,
        Y_Train,
        cv=kfold)

# Linear Discriminant Analysis
plot_learning_curve(
        gsLDA.best_estimator_,
        "Linear Discriminant Analysis learning curves",
        X_Train,
        Y_Train,
        cv=kfold)

""" ENSEMBLE """
# Ensemble Learners to: lower error and overfitting

# We first wish to compare how often our models predict the same thing

# Dropping PassengerId from predictor variables (could add as index)
X_Test = df_test.drop(["PassengerId"], axis=1)

# Creating Pandas dataframes of our models' predictions
test_pred_GB = pd.Series(GBC_best.predict(X_Test), name="GB")
test_pred_NN = pd.Series(NN_best.predict(X_Test), name="NN")
test_pred_LR = pd.Series(LR_best.predict(X_Test), name="LR")
test_pred_SVM = pd.Series(SVMC_best.predict(X_Test), name="SVM")
test_pred_LDA = pd.Series(LDA_best.predict(X_Test), name="LDA")

# Concatenating our models predictions
ensemble_results = pd.concat([test_pred_GB,test_pred_NN,test_pred_LR,test_pred_SVM, test_pred_LDA],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)

# ENSEMBLE MODEL
voting_model = VotingClassifier(
        estimators=[('gbc', GBC_best), ('nn', NN_best),
                    ('lr', LR_best), ('svm',SVMC_best),('lda',LDA_best)], 
        voting='soft', 
        n_jobs=4)

voting_model = voting_model.fit(X_Train, Y_Train)


""" Prediction """

# Getting the predictions of survived from our best GBC model
df_test["Survived"] = voting_model.predict(X_Test)

# Creating a solution dataframe
solution = df_test.loc[:,["PassengerId","Survived"]]

# Saving the solution dataframe to a csv
solution.to_csv("voting_Solution.csv", index=False)





# Getting the predictions of survived from our best GBC model
df_test["Survived"] = GBC_best.predict(X_Test)

# Creating a solution dataframe
solution = df_test.loc[:,["PassengerId","Survived"]]


# Getting the predictions of survived from our best NN model
df_test["Survived"] = NN_best.predict(X_Test)

# Creating a solution dataframe
solution = df_test.loc[:,["PassengerId","Survived"]]


# Getting the predictions of survived from our best LR model
df_test["Survived"] = LR_best.predict(X_Test)

# Creating a solution dataframe
solution = df_test.loc[:,["PassengerId","Survived"]]


# Getting the predictions of survived from our best SVM model
df_test["Survived"] = SVMC_best.predict(X_Test)

# Creating a solution dataframe
solution = df_test.loc[:,["PassengerId","Survived"]]


# Getting the predictions of survived from our best LDA model
df_test["Survived"] = LDA_best.predict(X_Test)

# Creating a solution dataframe
solution = df_test.loc[:,["PassengerId","Survived"]]


# Saving the solution dataframe to a csv
solution.to_csv("GBC_Solution.csv", index=False)


"""
END
"""