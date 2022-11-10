#!/usr/bin/env python
# coding: utf-8

# 
# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">1. Dataset (including citation of source and acknowledgment)</p>

# **Dataset Acknowledgment:**
# 
# UCI, 2014:German Credit Risk, 8.24. Database: Open Database, access date (31-10-2022), URL: https://www.kaggle.com/datasets/uciml/german-credit

#     - Build and train classification and/or regression models from the dataset in any suitable programming environment of your choosing (e.g., MATLAB) using three machine learning techniques of your choice.
# 
#     - Justify the rationale behind the choice of your dataset, machine learning techniques, and programming environment.
#     
#     - Compare and contrast the performance of the three machine learning techniques in terms of prediction or validation accuracy, training time, prediction speed, R-squared values, MSE values, and transparency (as may be applicable).
#     
#     - Analyse the error matrices, the ROCs (and AUCs) for all three methods (as may be applicable).
#     
#     - Comment on how the hyperparameters (if any) are tuned or optimized (if applicable) to enhance the built/trained models.
#     
#     - Submit a report showing the work carried out. Report like what u do after every module.

# ## Insightful approach to assist me while building my model
#     
#     - Is the data normally distributed or skewed?
#     - What's the gender characteristics?
#     - Does the Age good indicator to measure the Risk Level vs Credit amount?
#     - What's the correlation between the features of the dataframe?

# | Variables | Detailed Information |
# | -: | -: |
# | Age | Numeric Value |
# | Sex | male, female |
# | Job | 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled |
# | Housing | own, rent, or free |
# | Saving accounts | little, moderate, quite rich, rich |
# | Checking account | Numeric Value (in DM - Deutsch Mark) |
# | Credit amount | Numeric Value |
# | Duration | Numeric Value |
# | Purpose | car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others |

# 
# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">2. Definition and Justification of Problem According to the Dataset: Classification Problem and/or Regression Problem</p>

# Defination of the classification problem according to the dataset:
# 
# We have 1000 observations for customers and we want to classify the cutsomers based on dependent variable, so the bank will be able to assess the risk of the applications. We want to identify which users have higher risk level than normal based on their data [Age, Gender, Job Skills, Housing, Saving account, Checking account, Duration, Purpose of the loan].
# 
# My approach is the following:
# 
#     1. Explore the raw data then convert it into a dataframe.
#     
#     2. Wrangle the dataframe, and investigate the missing data, outliers, highlight the statistical inputs.
#     
#     3. Data Exploration; create essential graphs to identify the characteristics of each variable in the dataframe.
#     
#     4. Handle the missing data, usually it can be executed by using one of the imputation methods where to replace the missing variables with (mean > if the data is normally distributed - or mode/median > if the data is skewed)
#     I have decided to use machine learning in identifying the missing variables, beacuse it's more accurate and I will have the chance to implement what I have learned in this course. I couldn't done this step alone and I have included the books I have used in my approach as well as the academic papers from IEEE Xplore in the bottom of this file.
#     
#     5. Classification three techniques which are [Logistic Regression - Random Forest - Support Vector Machines] including the statistical aspect behind it.
#     
#     6. Comaprison between the three techniques where I use different methods such as ROC Curve, Classification Report, Confusion Matrix {Error Matrix}, AUC - Precision - Recall Ratios.
#     
#     7. Final conclusion.

# In[1]:


# import the required libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


# import the dataset:

df = pd.read_csv("german_credit_data.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)
df.head(2)


# In[3]:


df.info()


# In[4]:


round(df.describe(),2)


# In[5]:


round(df.corr(),2)


# 
# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3. Data Pre-processing and Feature Extraction: Identification of Predictors, Categories and Targets, Handling Noisy Data and Missing Data, Others</p>

# **Summarized Decription the steps here (I will explain them accordingly down below step by step)**
#     
#     - Check the missing data then replace the NaN values with Zero in order to fill them later properly.
#     - Change the D-type of the variables since most of them are labeled with "object" type.
#     - Label Encoding the variables such as "Gender & Job"

# 
# ### <p style="background-color:#FA8072; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.1 Checking the missing data then replace them with zeros</p>

# In[6]:


#Checking the NA values in the dataset

df.isna().sum()


# In[7]:


# replace NA with zeros

df['Saving accounts'] = df['Saving accounts'].fillna(0, inplace=False)
df['Checking account'] = df['Checking account'].fillna(0, inplace=False)


# 
# ### <p style="background-color:#FA8072; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.2 Change the D-type for Executing Feature Engineering Technqiues</p>

# In[8]:


# Convert the next variables into string type in order to apply Label Encoding later

df['Checking account'] = df['Checking account'].astype('string')
df['Saving accounts'] = df["Saving accounts"].astype("string")
df["Purpose"] = df["Purpose"].astype("string")


# 
# ### <p style="background-color:#FA8072; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.3 Implement Feature Engineering (Label Encoding) Technique</p>

# Label Encoding (LE) is an insipiring method to convert the variables into numeric variables but in order to do that, first we need to understand the variables dimension and values.
# 
# 
#     - Odinarl Data Type: BSc - Msc - PHD "more like ranking"
# 
#     - Nominal Data: Gender Binary Data: Yes/No
# 
# **So in the next section all the variables will be converted into numeric values using LE method.**
# 
# 
# Reference:
# 
# B. -B. Jia and M. -L. Zhang, "Multi-Dimensional Classification via Decomposed Label Encoding," in IEEE Transactions on Knowledge and Data Engineering, doi: 10.1109/TKDE.2021.3100436.

# In[9]:


# Since Sex variable & Housing are ordinarl variables then I will apply Encoding Labelling technique on both of
#them, because it sutis them best

le = preprocessing.LabelEncoder()

df['Housing'] = le.fit_transform(df['Housing'])
df["Gender"] = le.fit_transform(df["Sex"]) 
df['Checking account'] = le.fit_transform(df['Checking account'])
df['Saving accounts'] = le.fit_transform(df['Saving accounts'])


# In[10]:


# 1 for male & 0 for female as I have generate this code earier on the Sex variable

df["Gender"].value_counts()


# In[11]:


df["Sex"].value_counts()


# In[12]:


df.head(2)


# In[13]:


df.info()


# ### The purpose column in df dataset annotation is as the following:
# 
# 

# In[14]:


df["Purpose"].value_counts()


# In[15]:


# Drop the Sex Variable since I have Gender variable with 0 & 1.

df.drop("Sex", axis = 1, inplace = True)


# 
# ### <p style="background-color:#FA8072; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.4 Essential Explorations and Their Interpretations</p>

# In[16]:


# Plotting the histogram

df.hist(figsize = (12,12), color = "tomato")


# ## Interpretation:
# 
#     - From the histogram, we can see that many variables are right skewed.
#     - The variables with right skew are: Age, Saving accounts, Credit amount, Duration. That gives us hints to not depend on the mean but to focus more on the median and mode, since the mean value might make the final decision quite bias.
#     
#     - We have more date records related to the men than women. Based on the counting of the Gender variable.
#     
#     - Missing values in the checking_account variable are more than the rest of the correct values. It's a serious problem I need to pay a close attention to it.
#     
#     - Skilled individuals are the most figure welling to apply for loans, that's good indicator for the risk assessment that I will create later on.

# 
# ### <p style="background-color:#FA8072; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.5 Checking the outliers visually then statistically</p>

# In[17]:


# Creating the function to plot the numeric variables using Box Plot.

def box_plot(arg1, arg2, arg3, arg4):
    # Setting the graphs to be 4 plots at the same time
    fig, axes = plt.subplots(ncols=4, figsize=(20,15))
    
    # Measuring the correlationship between each variable and the Gender type:
    sns.boxplot( data=df, x='Gender', y=arg1,palette="Set3", orient='v', ax=axes[0])
    sns.boxplot( data=df, x='Gender', y=arg2, palette="Set3", orient='v', ax=axes[1])
    sns.boxplot( data=df, x='Gender', y=arg3, palette="Set3", orient='v', ax=axes[2])
    sns.boxplot( data=df, x='Gender', y=arg4, palette="Set3", orient='v', ax=axes[3])
    
    # Creating the title for the 4 graphs
    fig.suptitle('Gender Characteristics with Numeric Variables', fontsize = 20)
    
    # Highlighting the fontsize
    axes[0].set_ylabel(arg1,fontsize = 20)
    axes[1].set_ylabel(arg2,fontsize = 20)
    axes[2].set_ylabel(arg3, fontsize = 20)
    axes[3].set_ylabel(arg4, fontsize = 20)

    plt.show()


# In[18]:


box_plot("Age","Checking account","Credit amount","Duration")


# ## Interpretation:
# 
#     - Graph (1) = Gender vs Age =
#         - For men: we see that the average age that seek loans is above 35 years.
#         - For women: we see that the average age that seek loans is close to 30.
#         - We have outliers both sides and in the next section, we will highlight them in more details.
#         
#     - Graph (2) = Gender vs Checking_account = 
#         - For both men and women the mean value is 1.
#         
#     - Graph (3) = Gender vs Credit amount =
#         - For men: the avergae credit amount is 2500 unlike the women average credit amount is close 2000
#         - As we can see there is a unusal record needs to be invesitgated in the women box plot the record is above 17500. So in the next section, we will hgihlight the value with respect to the other observations in the dataframe.Then we will create a range of other individuals that have similar age, credit amount to decide whether to delete it or keep it.
#         
#     - Graph (4) = Gender vs Duration =
#         - For both women and men they have the avergae baseline for Duration which close to 20.
#         - Then again, unusual record from the men side that's above 70. And I will invesitgate that one too.

# In[19]:


# Checking the first graph [0 0] from the previous Box plot.

df[df["Age"] > 70]


# 
# **<pre style="color:purple;">Since this individual (index:756) is 74 years old, no job, with significant low credit amount. I have decided to eliminate this possible wrong typo from the system.<br>**

# In[20]:


# Drop the unusual record, since this individual is 74 years old, no job, with significant low credit amount.

df = df.drop(756).reset_index()
df.drop("index", axis=1,inplace=True)


# In[21]:


df.head(2)


# 
# **<pre style="color:purple;">Based on the previous Box plot, we can see a possible outlier, as there are a record above 17500 in the credit amount. So first I will investigate this observation then invesitgate the range of these records between the age 30 to 40 to see if that's a rational trend. Or it's insignificant to have such an observation.<br>**

# In[22]:


# Checking the possible outlier:

df[df["Credit amount"] > 17500]


# In[23]:


# Check the observation range, meaning check other individuals in the same age pool, 
#have multiple job skills and similar credit amount.

df[(df["Age"] > 30) & (df["Age"] < 40) & (df["Credit amount"] > 7500)]


# In[24]:


df = df.drop(914).reset_index()
df.drop("index", axis=1,inplace=True)


# 
# **<pre style="color:purple;">For a woman with such a high credit amount of 18424, and her saving accounts is moderate, and she is a very high skilled and she's not young, so my decision is to delete her record. Especially that after I have filtered the Age variable between 30 and 40 and credit amount above 7500. We can see that there are plenty of individuals have moderate credit records and they are close in age and job skills.<br>**
# 

# In[25]:


df[(df["Duration"] > 55) & (df["Age"] < 30)]


# **<pre style="color:purple;">After creating a range for individuals between the age 20 to 30, and have Duration period of more than 55, we can see that the unusual observation fit singificently among them, so I can't exclude the value.<br>**
# 

# 
# ### <p style="background-color:#FA8072; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.6 Essential Explorations and Their Interpretations</p>

# In[26]:


# Heatmap

plt.figure(figsize = (12,10))
sns.heatmap(df.corr(), annot = True, cmap ="Blues")


# ## Interpretation:
# 
#     - Heatmap represents the correlation between the variables of the dataframe.
#     - Some variables have positive/negative correlation with each other such as:
#         - Age: Positive correlation with Gender and negative correlation with Housing.
#         - Job: Positive correlation with Credit_amount then Duration.
#         - Housing: Negative correlation with Age and Job skills.
#         - Saving accounts: Negative correlation with credit amount.
#         - Credit amount: Strong positive correlation with Duration and positive correlation with Job and negative correlation with Saving accounts.
# 
# 
#     - These relationships will help me later while approaching my analysis.

# In[27]:


# Highlight the value counts in order to set the labels.

df["Job"].value_counts()


# In[28]:


# Plotting the Pie chart

df["Job"].value_counts().plot.pie(labels = ["Skilled", "Unskilled and Resident","Highly Skilled", "Unskilled and Non-resident"], figsize = (12,12), colors = ["#00A9DC", "#B01E8D", "#FDB812"])

plt.title("Job Distrbution Skills", fontsize = 14)
plt.ylabel("Counting Job Skills for Employees", fontsize = 14)

plt.show()

# https://www.w3schools.com/python/matplotlib_pie_charts.asp


# In[29]:


# Plotting the Pie chart

df["Purpose"].value_counts().plot.pie(figsize = (12,12), colors = ("coral", "cyan", "bisque", "limegreen", "lavender", "plum","lightskyblue", "yellow"))

plt.title("Purpose Distrbution", fontsize = 14)

plt.show()

# https://www.w3schools.com/python/matplotlib_pie_charts.asp


# ## Interpretation:
# 
#     - From the previous two pie plot, we conclude that for the Job skills applications we see the most visiable feature is skilled individuals followed by the Non-skilled but resident feature.
#     - When it comes to the Purpose the loans for; we see the most visiable reason is for car, then radio/TV then furniture/equipment.

# In[30]:


# Plotting the bar plot

df.groupby("Job")["Credit amount"].sum().plot.bar(figsize = (10,8), color = ["#DA70D6", "#ffcad9", "#20b2aa", "#e2c4f2"])

# Make sure it's well-interrpreted

plt.title("Measuring the Job Type per Credit Amount in Millions", fontsize=14)
plt.ylabel("Total Credit Amount", fontsize = 12)
plt.xlabel("Job Type", fontsize=12)

plt.xticks([0,1,2,3],["Unskilled and Non-resident", "Unskilled and Resident", "Skilled","Highly Skilled"], rotation=20, fontsize=10)
plt.grid(axis="x")
plt.legend()
plt.show()

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html


# ## Interpretation:
# 
#     - We can identifiy that the total credit amount for skilled people is close to 2 millions. Which might be related to their jobs and salaries followed by the highly-skilled. That gives us insight that the education variable if it was exist is a direct reason on the total amount. We could have created a hypothesis testing to check this potential before we create the classification models.

# 
# ### <p style="background-color:#FA8072; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.7 Use KNN (Regression) in Predicting the Missing Values for Saving_accounts & Checking_accounts</p>

# ## Detailed Approach on handling the missing values using KNN Regression:
# 
# K-Nearest Neighbor regressor is a technique can be used as an imputation method to predict the missing values. It's executed by measuring the ditsance between every data node and the other then pick the shortest distance and after calculating the distance matrix using the next equation:
# 
# 
# <h2><center>$d_{ij}=\sqrt{\sum_{y=1}^{p}(x_{iy}^{\prime}-x_{jy}^{\prime})^{2}}$</center></h2>
# 
# 
# Then computing the missing valuse under Y_prediction label using the next equation:
# 
# <h2><center>$V_{ij}=\frac{1}{k}\sqrt{\sum_{i=1}^{k}w_{i}x_{ij}}$</center></h2>
# 
# The detailed mathmatical interpretation is descibed in well-detailed in the IEEE paper [1,2].

# ### <pre style="color:purple;">Measuring the ratios of the missing values to identify the effect on the overall dataframe:<br>

# In[31]:


#Ratio of mussing data fro Saving accounts & Checking accounts

print("The Ratio of Missing Values in the Saving Accounts Variable is = ", (df["Saving accounts"] == 0).sum()/1000, "* 100")
print("The Ratio of Missing Values in the Checking Accounts Variable is = ", (df["Checking account"] == 0).sum()/1000, "* 100")


# ## Interpretation:
# 
# #### <pre style="color:blue;">Then the usual imputation technique for handling the missing data can cause bias or impact the accuracy and precision of my model later on. Then I will use Linear Regression to predict the missing values as shown down below:<br>
# 

# **First, drop the Purpose variable since it's categorical column, and it will generate an error later on.**

# In[32]:


df.drop("Purpose", axis=1, inplace=True)


# **Second, split the dataframe based on the "Saving Accounts" variable the values = 0 which are the missing variables, I will label them under X_test_saving_data. And the normal variables that represent correct data recording will be under the X_train_saving_data.**

# In[33]:



X_test_saving_data = df[df["Saving accounts"] == 0]
X_test_saving_data.head(2)


# In[34]:


# Checking the shape of X_test

X_test_saving_data.shape


# In[35]:


# Wanted to make sure that the counting is accurate and I didn't miss any values.

1000-183


# In[36]:


# Exclude missing values from the training sub-set

X_train_saving_data = df.loc[~((df['Saving accounts'] == 0))]
X_train_saving_data.head(2)

# https://stackoverflow.com/questions/49841989/python-drop-value-0-row-in-specific-columns


# In[37]:


# Checking the shape of X_train

X_train_saving_data.shape


# In[38]:


# Use the X_train sub-set to highlight the values of y_train

y_train_saving_data = X_train_saving_data['Saving accounts']
y_train_saving_data.head(2)


# In[39]:


# Checking the shape of y_train

y_train_saving_data.shape


# 
# ### <p style="background-color:#db7093; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.7.1 Building the KNN model for the Missing Values in the Saving Account</p>
# 
# **I have created 4 groups since "Saving_account" variable contains 4 labels [Little, Moderate, Quite Rich, High Rich]**

# In[40]:



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)
  
knn.fit(X_train_saving_data, y_train_saving_data)


# In[41]:


# Generate the prediction values

y_predict = knn.predict(X_test_saving_data)


# In[42]:


# Output the values

print(y_predict)


# In[43]:


# Check the shape before concatenate the whole sub-set

y_predict.shape


# In[44]:


# Concatenate the values back together

X_test_saving_data["Saving accounts"] = X_test_saving_data.loc[X_test_saving_data["Saving accounts"] == 0, "Saving accounts"] = y_predict

#https://stackoverflow.com/questions/61238384/replacing-values-in-pandas-dataframe-column-with-same-row-value-from-another-col


# In[45]:


# Check the printed values

X_test_saving_data.head(2)


# In[46]:


# Make sure I have omitted the 0 values which represent the missing values

X_test_saving_data["Saving accounts"].unique()


# In[47]:


X_test_saving_data.head(2)


# In[48]:


# Counting the unique values in the column

X_test_saving_data["Saving accounts"].value_counts()


# In[49]:


X_train_saving_data["Saving accounts"].value_counts()


# In[50]:


print(173+603)
print(7+103)
print(2+63)
print(1+48)


# In[51]:


df00 = pd.concat([X_train_saving_data, X_test_saving_data], 0, ignore_index=True)


# https://stackoverflow.com/questions/46269804/concatenating-dataframes-on-a-common-index


# In[52]:


df00.head(2)


# In[53]:


df00["Saving accounts"].value_counts()


# 
# ### <p style="background-color:#db7093; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.7.2 Building the KNN model for the Missing Values in the Checking Account</p>
# 
# 
# **I have created 4 groups since "Saving_account" variable contains 3 labels.**

# In[54]:


### Let's do the Checking account too:

df["Checking account"].unique()


# In[55]:


# Highlight the missing values in the column Checking account

X_test_saving_data0 = df[df["Checking account"] == 0]
X_test_saving_data0.head(2)


# In[56]:


# Check the shape of the new sub-set, as it's usually generate an error in the steps followed if it's done incorrectly

X_test_saving_data0.shape


# In[57]:


# Hightlight the non-zero values in the checking account

X_train_saving_data0 = df.loc[~((df['Checking account'] == 0))]
X_train_saving_data0.head(2)


# In[58]:


# Check the sub-set shape

X_train_saving_data0.shape


# In[59]:


# Generate the y-train sub-set

y_train_saving_data0 = X_train_saving_data0['Checking account']
y_train_saving_data0.head(2)


# In[60]:


# Check the sub-set shape

y_train_saving_data0.shape


# In[61]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
  
knn.fit(X_train_saving_data0, y_train_saving_data0)


# In[62]:


# Creating the missing values as y_pred

y_pred = knn.predict(X_test_saving_data0)


# In[63]:


print(y_pred)


# In[64]:


# Check the sub-set shape

y_pred.shape


# In[65]:


X_test_saving_data0["Checking account"] = X_test_saving_data0.loc[X_test_saving_data0["Checking account"] == 0, "Checking account"] = y_pred


# In[66]:


X_test_saving_data0["Checking account"].unique()


# In[67]:


X_test_saving_data0["Checking account"].value_counts()


# In[68]:


X_train_saving_data0["Checking account"].value_counts()


# In[69]:


print(210+274)
print(176+269)
print(8+63)


# In[70]:


df0 = pd.concat([X_train_saving_data0, X_test_saving_data0], 0, ignore_index=True)


# In[71]:


df0["Checking account"].value_counts()


# ### <p style="background-color:#db7093; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.7.3 Concatenate the Predicted Values from Both the Saving_accounts & Checking_accounts in the Original Dataframe "df"</p>
# 

# In[72]:


df["Checking_accounts"] = df0["Checking account"]
df["Saving_accounts"] = df00["Saving accounts"]


# In[73]:


df.head(2)


# In[74]:


# Drop the original old columns

df.drop(["Saving accounts", "Checking account"], axis = 1, inplace = True)


# In[75]:


# Checking the D-type of the columns

df.info()


# In[76]:


# Double check if there's any miss-calculations

df.isna().sum()


# ## mission accomplished :)

# ### <p style="background-color:#FA8072; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">3.8 Create Clustering Analysis to Create Risk Level</p>
# 
# 

# In[ ]:





# In[77]:


x = df.iloc[:, :-1].values
y = df.iloc[:,-1].values

#checking the shape of the input and output features

print('Shape of the input features:', x.shape)
print('Shape of the output features:', y.shape)


# In[78]:


#spliting the features into train & test sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.30, random_state=100)


# In[79]:


#checking the shape of the training & test sets
print('Shape of the training patterns:', x_train.shape,y_train.shape)
print('Shape of the testing patterns:', x_test.shape,y_test.shape)


# In[80]:


from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    cs.append(kmeans.inertia_)
    
plt.figure(figsize = (20,15))

plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# In[81]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3,random_state=0)

kmeans.fit(x)

labels = kmeans.labels_

# check how many of the samples were correctly labeled

correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# 
# ### <p><strong>As we can see here the best number of clusters is 3 as it gives the highest accuracy possible. The best optimal K = 3</strong></p>
# 

# In[82]:


kmeans = KMeans(3)
kmeans.fit(x)


# In[83]:


identified_clusters = kmeans.fit_predict(x)


# In[84]:


identified_clusters = pd.DataFrame(identified_clusters)
identified_clusters.value_counts()


# In[85]:


identified_clusters.columns = ["Risk_Level"]


# In[86]:


identified_clusters.head(2)


# In[87]:


df["Risk Level"] = identified_clusters["Risk_Level"]


# In[88]:


df.head(2)


# In[89]:


#Re-order the columns in the df data frame

df = df[["Age", "Gender", "Job", "Housing", "Saving_accounts", "Checking_accounts", "Credit amount", "Duration",
        "Risk Level"]]


# **Create Age Category using the recent Risk Level variable**

# In[90]:


data = df

# ["Youtth 19-29", "Adult 30-40", "MiddleAge 41-50", "Old 51-60", "VeryOld 61-76"]


# In[91]:


bins = [19,29,40,50,60,76]
labels = ["19-29", "30-40", "40-50", "51-60", "60-76"]
data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
data.head(2)


# In[92]:


data.groupby(["AgeGroup", "Risk Level"], sort = True)["Credit amount"].sum().plot.bar(figsize= (14,8), color = "teal")
plt.ylabel("Total Credit Amount", fontsize = 12)
plt.xlabel("Age & Risk Level", fontsize = 12)
plt.title("Total Credit amount per Age Group & Risk Level", fontsize = 14)

plt.legend(fontsize=12)
plt.grid(axis="y")
plt.show()


# ## Interpretation:
# 
#     - I have used the recent "Risk Level" variable, in order to categorize the age values into decent groups and measure the interaction between the age groups and credit amount. To check the traffic of the applications.
#     
#     - We can highlight main features in the previous bar plot as following:
#     
#         - Low Risk Level represents by 0: Individuals in the age group (30-40) recorded total credit amount of 500k, and in the age group (19-29) they recorded a total credit amount close to 500k. When it comes to the age group (40-50) recorded a toal credit amount above 200k.
#         
#         - Medium Risk Level represents by 1: According the previous plot the medium risk has decent ratios for all the age groups, maybe the peak point is for the age group (30-40) that recorded above 200k.
#         
#         - High Risk Level represents by 2: Age group (30-40) recorded total credit amount above 400k, and age group (19-29) recorded total credit amount above 300k. We might want to investigate the last section in more depth to see the other variables of the dataframe. As this category impact directly the decision of the stakeholders.
#         

# In[ ]:





# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Rationale Informing the Selection or Choice of the Three Machine Learning Techniques or Methods to Build and Train Models to Address the Problem</p>

# # The choosen classification techniques are:
#     
#     1. Logistic Regression (LR)
#     2. Support Vector Machines (SVM)
#     3. Random Forest (RF)
#     
# ## Those are based on multiple reasons:
# 
#     - Reason (1): I want to build up a model with different approaches; LR is a linear model and this type can use a hyperplane (Forward line) to divide the classes visually. On the other hand, SVM is the goldent ticket as you will see later it can be a linear model or a non-linear model. It works smoothly with this dataset and generates the highest rates in accuracy, confusion matrix and other matrix. Finally, the RF is a non-linear method and I want to gives this dataset different domains to execute useful insights upon.
#     
#     - Reason (2): LR can't give the best results with noise data, and this dataset is small not large (doesn't have millions of records), and some imbalanced attributes that's why trying different techniques can enhance my overall model.
#     
#     - Reason (3): The hyperparameters of this dataframe are optimized, since I have converted the categorical variables using lebel encoding to improve the prediction rates. Hyperparamter optimization is the method of searching for the right features to maximize tge performance of the data in a rational period of time. 
#     
#     - Reason (4): Choosing RF, I wanted to have a straight-forward model with (yes/no) scenarios. In practical/business wise, stakeholders need answers like if the credit amount is under specific value, should we consider this user's application as high risk or low risk. And I have created a tree plot to visualize such scenarios.
#     
#     - Reason (5): Random Froest doesn't need any feature scalling unlike logistic regression as well as SVM.
#     
#     - Reason (6): Feature importance plays vital role in RF technique, and later on it will boost the accuracy rate to 100% 

# 
# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Model Performance Evaluation Using the Logistic Regression Machine Learning Technique</p>

# ## Main Characteristics:
# 
# Logistic Regression divides into binary classification and multi-label classification (Low Risk Level - Medium Risk Level - High Risk Level), and the second one is what I will implement later on in this file.
# 
# One of the main characteristic in LR is that based on probability rules and it uses Sigmiod function to not be restricted in the range of 0 and 1.
# 
# ## How it works:
# 
# Simiply, linear regression rules apply, so we have a dependent variable and multiple independent variables and we will classify/predict the unique values of the dependent variable using the independent variables.
# 
# ## What's the statistical approach based on:
# 
# The following is the LR equation:
# 
# <h2><center>$sigma(z)=\frac{1}{1+e^{-z}}$</center></h2>
# 
# The Sigmiod equation is: (where n>=1)
# 
# <h2><center>$sigma(z)=\frac{1}{1+e^{-nz}}$</center></h2>
# 
# However, while optimization approach we have to make the error function (it's the function of regression coefficients) reach the smaller amount as it reflects on iteration numbers and the runtime of the whole algorithms.
# 
# The error function is:
# 
# <h2><center>$Y= \frac{1}{2}(y^{\prime}-y)^{2}$</center></h2>
# 
# [6]
# 
# ## Limitation of this method:
# 
# - LR assumes there's no multicollinearity in the altercation of independent features.
# - This technique is ideal for non-noisy data, and the improved methods in deep learning or neural network can overcome this classic mdethod.
# 
# 

# 
# ## <pre style="color:purple;">Splitting the dataframe into train & test subsets for LR<br>

# In[93]:


# selecting features:


X = df[["Age", "Gender", "Job", "Housing", "Credit amount", "Duration"]]
y = df["Risk Level"]


# In[94]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=100)


# In[95]:


# Import the requiored libraries

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[96]:


from sklearn.linear_model import LogisticRegression

Logistic_classifier_clf = LogisticRegression()
print(Logistic_classifier_clf.fit(X_train, y_train))

Logistic_classifier_predict = Logistic_classifier_clf.predict(X_test)


# In[97]:


print("The Accuracy Percentage for Logistic Regression Prediction is: ", round(accuracy_score(Logistic_classifier_predict, y_test),5)*100, "%")


# **92.40% is good indicator but we will try to improve this value before creating the ROC Curve.**

# 
# ## <pre style="color:purple;">Creating the Classification Report for LR<br>

# In[98]:


print("The Logistic Regression Classification Report")
target_names = ['Low Risk Level', "Medium Risk", 'High Risk Level']
print(classification_report(Logistic_classifier_predict, y_test, target_names=target_names))


# **As we can see here that the precision for Low Risk Level is great ratio unlike the Medium & High Risk Level. And that's the downside of the LR, it can't handle the noise and these values might cause overfitting in the testing subset. Overfitting means that the model is performing very well on the training dataset and not too-well on the testing dataset. Since LR is a linear model, we will see the same impact on the other 2 classification models.**

# 
# ## <pre style="color:purple;">Creating the Confusion Matrix for LR<br>

# In[99]:


LR_Confusion_matrix = confusion_matrix(Logistic_classifier_predict, y_test)
print(LR_Confusion_matrix)


# In[100]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(LR_Confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Logistic Regression', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label');


# **As we can see here: TP (True Positive) are 171 - 11 - 49.**
# 

# 
# ## <pre style="color:purple;">Creating the ROC Curve for LR<br>

# In[101]:


#Calculate the y_score
LR_y_score = Logistic_classifier_clf.fit(X_train, y_train).predict_proba(X_test)


# In[102]:


from sklearn.preprocessing import label_binarize

#Binarize the output
LR_y_test_bin = label_binarize(y_test, classes=[0,1,2])
n_classes = LR_y_test_bin.shape[1]


# In[103]:


from sklearn import metrics

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
  fpr[i], tpr[i], _ = metrics.roc_curve(LR_y_test_bin[:, i], LR_y_score[:, i])
  plt.plot(fpr[i], tpr[i], color='coral', lw=2)
  print('AUC for Class {}: {}'.format(i+1, metrics.auc(fpr[i], tpr[i])))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest')
plt.show()


# **As we can see here features have been improved from 77% for Medium Risk to 99%.**

# In[104]:


# Create a scatter chart to see the baseline for each risk level

sns.scatterplot(x="Credit amount", y="Risk Level", data=df, color='teal')


# In[ ]:





# 
# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Model Performance Evaluation Using the Support Vector Machines (SVM) Machine Learning Technique</p>

# 
# ## How SVM works in classification:
# 
# 
# SVM is a supervised machine learning algorithm that can be implemented for classification problems or regression problems. It's based on a method known as Linear Kernel which transforms the data values and looks for an optimal boundary between the potential outcomes. [3,4,5]
# 
# ## Statistical Equation behind this technique:
# 
# <h2><center>$f(x)=Sgn\left[\sum\limits_{i=1}^{n}\alpha_{i}y_{i}K(x_{i},x)+b\right]$</center></h2>
# 
# 

# In[105]:


from sklearn.svm import SVC


SVM_classifier_clf = SVC(kernel='linear', probability=True)
print(SVM_classifier_clf.fit(X_train, y_train))


SVM_classifier_predict = SVM_classifier_clf.predict(X_test)


# In[106]:


print("The Accuracy Percentage for SVM Prediction is: ", round(accuracy_score(SVM_classifier_predict, y_test),5)*100, "%")


# In[107]:


print("The SVM Classification Report")
target_names = ['Low Risk Level', "Medium Risk", 'High Risk Level']
print(classification_report(SVM_classifier_predict, y_test, target_names=target_names))


# **As we can see here that the precision for Low Risk Level is 100% unlike the Medium Risk Level 93%. However the F1-score which represents the interaction between the precision and recall ratio is pretty high for all the risk levels and from the range 0 to 9. We have the highest F1-score for SVM technique. This report proves that SVM mechanisim interact very well with this dataset despite the imbalanced, nosiy attributes.**

# In[108]:



SVM_Confusion_matrix = confusion_matrix(SVM_classifier_predict, y_test)
print(SVM_Confusion_matrix)


# In[109]:




class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(SVM_Confusion_matrix), annot=True, cmap="flare" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for SVM', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label');


# **As we can see here: TP (True Positive) are 179 - 13 - 57.**

# In[ ]:





# In[110]:


#Calculate the y_score
SVM_y_score = SVM_classifier_clf.fit(X_train, y_train).predict_proba(X_test)


# In[111]:


from sklearn.preprocessing import label_binarize

#Binarize the output
SVM_y_test_bin = label_binarize(y_test, classes=[0,1,2])
n_classes = SVM_y_test_bin.shape[1]


# In[112]:


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
  fpr[i], tpr[i], _ = metrics.roc_curve(SVM_y_test_bin[:, i], SVM_y_score[:, i])
  plt.plot(fpr[i], tpr[i], color='coral', lw=2)
  print('AUC for Class {}: {}'.format(i+1, metrics.auc(fpr[i], tpr[i])))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for SVM')
plt.show()


# **As we can see here features for class 2 have been improved from 93% to 100%.**

# 
# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Model Performance Evaluation Using the Random Forest Machine Learning Technique</p>

# 
# ## How RF works in classification:
# 
# 
# SVM is a supervised machine learning algorithm and one of the best advanced techniques, big plus is that it's capable of handling very large raw dataset that doesn't have clear patterns with high accuracy. By creating number of decision trees in a short period of time for the training & testing subsets.
# 
# 

# In[117]:


# Building the RF model

from sklearn.ensemble import RandomForestClassifier

Random_Classifier_clf = RandomForestClassifier(n_estimators=100)

print(Random_Classifier_clf.fit(X_train, y_train))

Random_Classifier_predict = Random_Classifier_clf.predict(X_test)


# 
# ## <pre style="color:purple;">Implementing the Features Importance in Random Forest<br>

# In[118]:


feature_importance = pd.Series(Random_Classifier_clf.feature_importances_,
                               index=["Age", "Gender", "Job", "Housing", "Credit amount", "Duration"])

feature_importance = feature_importance.sort_values(ascending=False)
feature_importance = round(feature_importance*100,3)
feature_importance


# In[119]:


sns.barplot(x=feature_importance, y=feature_importance.index, palette = ("teal", "lightgreen","coral", "lightblue", "darkgrey", "white"))

plt.title("Modeling the Importance of Each Feature", pad=15, fontsize = 14)
plt.xlabel("The Importance Measurement", fontsize=14)
plt.ylabel("Feature", fontsize=14)

plt.grid(axis="x")
plt.show()


# ## Interpretation:
# 
#     - As we can see here that there are some crucial features (Credit amount & Duration) and other features we can eliminate them to improve our model.

# 
# ## <pre style="color:purple;">Redplot the Random Forest again after eliminating the unnecessary variables<br>

# In[120]:


# selecting features:


X = df[["Credit amount", "Duration"]]
y = df["Risk Level"]


# In[121]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=100)


# In[122]:


Random_Classifier_clf = RandomForestClassifier(n_estimators=100)


# **Predicting the model**

# In[123]:


Random_Classifier_clf.fit(X_train, y_train)

Random_Classifier_predict = Random_Classifier_clf.predict(X_test)


# **Measuring the accuracy of the model**

# In[124]:


print("The Accuracy Percentage for Random Forest Prediction is: ", round(accuracy_score(Random_Classifier_predict, y_test),5)*100, "%")


# 
# ## <pre style="color:purple;">Creating the Random Forest Tree after deploying the feature importance technique<br>

# In[125]:


from sklearn.tree import plot_tree

plt.figure(figsize=(60,30))
plot_tree(Random_Classifier_clf.estimators_[5],
         filled=True,feature_names=["CreditAmount", "Duration"],
         class_names=['Low Risk Level', "Medium Risk Level", 'High Risk Level'],
          rounded=True, proportion=False,precision=2)


# ## Interpretation:
# 
#     - I have attached the image of the Random Forest Tree down below in the conclusion sectio to zoom-in & out easily also it will be in the folder that I will share the link on the top as well for more guides.
#     
#     - As we can see here if we followed the tree branch for an example to simplify:
#         - Is the Duration <= 25 >> yes >> Is the Duration <= 13.50 << No >> Is the credit amount <= 3787.5 >> yes >> then it's Low Risk Level.
#         
#     - This method is absoutely insipiring to highlight the features of the dataframe and the nueric numbers associated with to use them later for further investiagtion (as an example investaiagte the High Risk Level tree branches that are associated with certain credit amount figure).

# 
# ## <pre style="color:purple;">Creating the Classification Report for RF<br>

# In[126]:


print("The Random Forest Classification Report")
print(classification_report(Random_Classifier_predict, y_test, target_names=target_names))


# **As we can see here that the precision for Low & High Risk Level is 100% unlike the Medium Risk Level is 86% which pretty low compared the other two. On the other hand, the recall ratio is significant since the recall represents the percentage between the values of true positive to the total number of positive values.**
# 

# 
# ## <pre style="color:purple;">Creating the Confusion Matrix for RF<br>

# In[128]:


RF_Confusion_matrix = confusion_matrix(Random_Classifier_predict, y_test)
print(RF_Confusion_matrix)


# In[129]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(RF_Confusion_matrix), annot=True, cmap="Blues" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label');


# 
# ## <pre style="color:purple;">Creating the ROC Curve for RF<br>

# In[130]:


#Calculate the y_score
RF_y_score = Random_Classifier_clf.fit(X_train, y_train).predict_proba(X_test)


# In[131]:


from sklearn.preprocessing import label_binarize

#Binarize the output
RF_y_test_bin = label_binarize(y_test, classes=[0,1,2])
n_classes = RF_y_test_bin.shape[1]


# In[132]:


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
  fpr[i], tpr[i], _ = metrics.roc_curve(RF_y_test_bin[:, i], RF_y_score[:, i])
  plt.plot(fpr[i], tpr[i], color='coral', lw=2)
  print('AUC for Class {}: {}'.format(i+1, metrics.auc(fpr[i], tpr[i])))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest')
plt.show()


# **As we can see here features have been improved from 86% for Medium Risk to 100%.**

# In[ ]:





# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Comparisons between all Machine Learning Techniques or Methods</p>

# | Risk Level Type | Classification Model | Accuracy | Precision | Recall | F1-score | True Positive (TP) |
# | -: | -: | -: | -: | -: | -: | -: |
# | Low Risk Level | Logistic Regression (LR) | 96.70% | 96% | 97% | 96% | 171 |
# | Medium Risk Level | Logistic Regression (LR) | 99% | 79% | 85% | 81% | 11 |
# | High Risk Level | Logistic Regression (LR) | 94.90% | 86% | 82% | 84% | 49 |
# | Low Risk Level | Support Vector Machines (SVM) | 100% | 100% | 100% | 100% | 179 |
# | Medium Risk Level | Support Vector Machines (SVM) | 100% | 93% | 100% | 96% | 13 |
# | High Risk Level | Support Vector Machines (SVM) | 99.98% | 100% | 98% | 99% | 57 |
# | Low Risk Level | Random Forest (RF) | 100% | 100% | 100% | 100% | 179 |
# | Medium Risk Level | Random Forest (RF) | 100% | 86% | 100% | 100% | 12 |
# | High Risk Level | Random Forest (RF) | 100% | 100% | 97% | 98% | 57 |

# 
# **<pre style="color:blue;">The main differences between the Logistic Regression, Support Vector Machines and Random Forest is that the LR is linear model sesitive to noisy, imbalanced data. LR pays maximum ratios in normally distributed datasets. Unlike the SVM and RF models both aren't sensitive for such imbalanced, nosiy data.<br>**
# 
# **<pre style="color:blue;">Additionally While creating the ROC Curve, I have refined the features of the model and it paied off as you can see int he previous table. When it comes to the RF model, I have highlighted the features importance before creating the plot tree.<br>**
# 
# **<pre style="color:blue;">Based on the table above, we can see whether Support Vector Machines or Random Forest classification methods, both handle the data values more significant than the Logistic Regression. The ratios for accuracy which represents how accurate the traing subset is capable of predicting the testing subset by using the same modeling, and precision which represents how precise the model of generating the same values over and over again while deploying, recall which represents the ratio true positive compared to the all true positive in the data sampling. All the ratios are impressive. The most flexible classification techniques in order are SVM then RF and finally LR.<br>**

# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Recommendations and Conclusions</p>

# Final Conclusion:
#     
#         - This dataset is rich in the observations and it might be so useful if there is a possability to add more records to create further analysis, such as hypothesis testing.
#         
#         - Where I have started in this dataframe was to create a new column to start classifying the variables in this dataframe. But to do that I had to refine my data values to make sure that there's no missing values/outliers/inconsistency. And the fastest, easiest way is to use imputation technique such as (mean, mode, median) however since the dataframe is skewed to the right I wanted more accurate precise results.
#         
#         - I have used machine learning to predict the missing values (K-Nearest Neighbor) method.
#         
#         - Then I have used clsutering (kmeans with distance calculations) to create a new column called Risk Level. Where I group all the observations in the dataframe with respect to this column. So this column becomes the dependent variable that depends on the original independent columns in the dataset.
#         
#         - Once I reach to this point, I have a solid dataframe to start my classification methods from Logistic Regression, Support Vector Machines and Random Forest.
#         
#         - Finally generating useful insight matrics to runa comparison between the different models. I have clarified the differences at the top while I explained why I chose those, then in every section.

# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">References</p>

# ## Academic Materials "IEEE Xplore & Books"
# 
# [1] S. Sen, M. N. Das and R. Chatterjee, "A Weighted kNN approach to estimate missing values," 2016 3rd International Conference on Signal Processing and Integrated Networks (SPIN), 2016, pp. 210-214, doi: 10.1109/SPIN.2016.7566690.
# 
# [2] Shahla Faisal, Gerhard Tutz, "Multiple imputation using nearest neighbor methods," Information Sciences, Volume 570, 2021, Pages 500-516, ISSN 0020-0255, doi: https://doi.org/10.1016/j.ins.2021.04.009.
# 
# [3] Yujun Yang, Jianping Li and Yimei Yang, "The research of the fast SVM classifier method," 2015 12th International Computer Conference on Wavelet Active Media Technology and Information Processing (ICCWAMTIP), 2015, pp. 121-124, doi: 10.1109/ICCWAMTIP.2015.7493959.
# 
# [4] J. Huang, J. Zhou and L. Zheng, "Support Vector Machine Classification Algorithm Based on Relief-F Feature Weighting," 2020 International Conference on Computer Engineering and Application (ICCEA), 2020, pp. 547-553, doi: 10.1109/ICCEA50009.2020.00121.
# 
# [5] Chen Junli and Jiao Licheng, "Classification mechanism of support vector machines," WCC 2000 - ICSP 2000. 2000 5th International Conference on Signal Processing Proceedings. 16th World Computer Congress 2000, 2000, pp. 1556-1559 vol.3, doi: 10.1109/ICOSP.2000.893396.
# 
# [6] X. Zou, Y. Hu, Z. Tian and K. Shen, "Logistic Regression Model Optimization and Case Analysis," 2019 IEEE 7th International Conference on Computer Science and Network Technology (ICCSNT), 2019, pp. 135-139, doi: 10.1109/ICCSNT47585.2019.8962457.

# ### Convert the Jupyter Notebook to PDF

# In[1]:


import pdfkit

options = {
'page-size': 'A4',
'margin-top': '0in',
'margin-right': '0in',
'margin-bottom': '0in',
'margin-left': '0in',
'encoding': "UTF-8",
'no-outline': None
}

# https://github.com/wkhtmltopdf/wkhtmltopdf/issues/2810
# https://pdfkit.org/docs/guide.pdf
# https://stackoverflow.com/questions/65991584/how-to-specify-pdf-page-size-using-pdfkit-in-python


# In[3]:


pdfkit.from_file('Rewan Emam - S21003259 -- ML ( Applied Machine Learning Project -- 24-10-2022).html', 
                 'Rewan Emam - S21003259 -- ML ( Applied Machine Learning Project -- 24-10-2022).pdf',options=options)


# In[ ]:




