#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries require for the project
# 
# 

# In[2]:


#libraries to help with reading and manipulating data
import pandas as pd
import numpy as np


# In[3]:


#libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Importing the Machine Learning models we require from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# In[5]:


# Importing the other functions we may require from Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder


# In[6]:


# To get different metric scores
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,plot_confusion_matrix,precision_recall_curve,roc_curve,make_scorer


# In[7]:


# Code to ignore warnings from function usage
import warnings;
import numpy as np
warnings.filterwarnings('ignore')


# # loading the dataset
# 

# In[8]:


hotel=pd.read_csv("hotel.csv")
hotel


# In[9]:


#viewing the first 5 rows of the dataset
hotel.head()


# In[10]:


#viewing the last 5 rows of the dataset
hotel.tail()


# In[11]:


# Copying data to another variable to avoid any changes to original data
data=hotel.copy()

#viewing the first 5 rows of the dataset
data.head()


# In[12]:


#Understand the shape of the data
data.shape

There are 36275 rows and 19 columns in the dataset.
# In[13]:


#Check the data types of the columns for the dataset
data.info()

Observations:
There are 36275 observations and 19 columns in the dataset.
All the columns have 36275 non-null values, i.e. there are no missing values in the data.
Booking_ID, type_of_meal_plan, room_type_reserved, market_segment_type, and booking_status are of object type while rest columns are numeric in nature.
# In[14]:


# checking for duplicate values
data.duplicated().sum()


# There are no duplicate data
# 

# # Dropping the unique values column
We will drop the Booking_ID column first before we proceed forward, as a column with unique values will have almost no predictive power for the Machine Learning problem at hand.
# In[15]:


data = data.drop(["Booking_ID"], axis=1)
data.head()


# # Exploratory Data Analysis
Question 1:
Check the statistical summary of the data
# In[17]:


data.describe().T


# observations:
# 
# The average price per room is 103.42 Euros with the lowest price being 0.0 Euros and the highest price being 540 Euros.There are few rooms which have price equal 0.
# 1 out of the 4 adult guest (4 is maximum number of adult guest) is a repeated guest.
# The average number of days between the date of booking and the arrival date is 85 days.
# 
Question 2: Univariate Analysis
Perform univariate analysis on the columns of this dataset

We will first define a hist_box() function that provides both a boxplot and a histogram in the same visual, with which we can perform univariate analysis on the columns of this dataset.
# In[18]:


# Defining the hist_box() function
def hist_box(data,col):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)})
    # Adding a graph in each part
    sns.boxplot(data[col], ax=ax_box, showmeans=True)
    sns.distplot(data[col], ax=ax_hist)
    plt.show()


# Question 2.1:¶
# Plot the histogram and box plot for the variable Lead Time using the hist_box function provided and write your insights
# 

# In[19]:


hist_box(data, "lead_time")


# Observations:
# The variable "lead time" have skewed distribution.
# The distribution for "lead time" is right skewed with outliers
# 
Question 2.2:
Plot the histogram and box plot for the variable Average Price per Room using the hist_box function provided and write your insights
# In[20]:


hist_box(data, "avg_price_per_room")

Observations:
The variable "avg_price_per_room" have some outliers to the right end.
The variable "avg_price_per_room" is positively skewed.
The higher the price of the room, the lower the number of guest that book such room.
# In[21]:


# checking for rooms with price equal to zero
data[data["avg_price_per_room"] == 0]

Observation:
There are few rooms with price equal to zero
In the market segment column, it looks like many values are complementary
# In[22]:


data.loc[data["avg_price_per_room"] == 0,"market_segment_type"].value_counts()

It makes sense that most values with room prices equal to 0 are the rooms given as complimentary service from the hotel.# free service from hotel.
The rooms booked online must be a part of some promotional campaign done by the hotel.
# In[23]:


# Calculating the 25th quantile
Q1 = data["avg_price_per_room"].quantile(0.25)

# Calculating the 75th quantile
Q3 = data["avg_price_per_room"].quantile(0.75)

# Calculating IQR
IQR = Q3 - Q1

# Calculating value of upper whisker
Upper_Whisker = Q3 + 1.5 * IQR
Upper_Whisker


# In[24]:


# assigning the outliers the value of upper whisker
data.loc[data["avg_price_per_room"] >= 500, "avg_price_per_room"] = Upper_Whisker


# Let's understand the distribution of the categorical variables
# Number of Children
# 

# In[25]:


sns.countplot(data['no_of_children'])
plt.show()

Most time, customers in the hotel do not come with children.

# In[26]:


data['no_of_children'].value_counts(normalize=True)

Customers were not travelling with children in 93% of cases.
There are some values in the data where the number of children is 9 or 10, which is highly unlikely.
We will replace these values with the maximum value of 3 children.
# In[27]:


# replacing 9, and 10 children with 3
data["no_of_children"] = data["no_of_children"].replace([9, 10], 3)

Arrival Month
# In[28]:


sns.countplot(data["arrival_month"])
plt.show(


# In[29]:


data['arrival_month'].value_counts(normalize=True)


# October is the busiest month for hotel arrivals followed by September and August.
# 
# Over 35% of all bookings, as we see in the above table, were for one of these three months.
# 
# Around 14.7% of the bookings were made for an October arrival.
# 
Booking Status
# In[30]:


sns.countplot(data["booking_status"])
plt.show()


# In[31]:


data['booking_status'].value_counts(normalize=True)


# 32.8% of the bookings were canceled by the customers.
# 
Encode Canceled bookings to 1 and Not_Canceled as 0 for further analysis
# In[32]:


data["booking_status"] = data["booking_status"].apply(
    lambda x: 1 if x == "Canceled" else 0
)

Question 3.1:
Find and visualize the correlation matrix using a heatmap and write your observations from the plot.
# In[33]:


cols_list = data.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(12, 7))
sns.heatmap (data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()

Average Price Per Room negatively correlate with Number of Previous Cancellation.
Booking Status negatively correlate with Arrival month.
Number of Bookings not Cancelled positively correlate with repeated quest.
Booking Status positively correlated with Lead Time.
Hotel rates are dynamic and change according to demand and customer demographics.¶
Let's see how prices vary across different market segments
# In[34]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="market_segment_type", y="avg_price_per_room", palette="gist_rainbow")
plt.show()

Rooms booked online have high variations in prices.
The offline and corporate room prices are almost similar.
Complementary market segment gets the rooms at very low prices, which makes sense.We will define a stacked barplot() function to help analyse how the target variable varies across predictor categories.
# In[35]:


# Defining the stacked_barplot() function
def stacked_barplot(data,predictor,target,figsize=(10,6)):
  (pd.crosstab(data[predictor],data[target],normalize='index')*100).plot(kind='bar',figsize=figsize,stacked=True)
  plt.legend(loc="lower right")
  plt.ylabel('Percentage Cancellations %')

Question 3.2:
Plot the stacked barplot for the variable Market Segment Type against the target variable Booking Status using the stacked_barplot function provided and write your insights.
# In[36]:


stacked_barplot(data,"market_segment_type","booking_status")

In Aviation Segment about 30% of bookings were cancelled.
No bookings made were cancelled in Complementary Segment.
About 35% of bookings made online were cancelled.
About 33% of bookings made offline were cancelled.
In orporate Segment about 10% of bookings were cancelled.Question 3.3:¶
Plot the stacked barplot for the variable Repeated Guest against the target variable Booking Status using the stacked_barplot function provided and write your insights
# In[37]:


stacked_barplot(data, "repeated_guest","booking_status")

About 35% of guest who visited the hotel before cancelled the booking for next visit/stay.
About 2% of repeated guuest cancelled the booking.Analyze the customer who stayed for at least a day at the hotel.
# In[38]:


stay_data = data[(data["no_of_week_nights"] > 0) & (data["no_of_weekend_nights"] > 0)]
stay_data["total_days"] = (stay_data["no_of_week_nights"] + stay_data["no_of_weekend_nights"])

stacked_barplot(stay_data, "total_days", "booking_status",figsize=(15,6))

The general trend is that the chances of cancellation increase as the number of days the customer planned to stay at the hotel increases.As hotel room prices are dynamic, Let's see how the prices vary across different months.
# In[39]:


plt.figure(figsize=(10, 5))
sns.lineplot(y=data["avg_price_per_room"], x=data["arrival_month"], ci=None)
plt.show()

The price of rooms is highest in May to September - around 115 euros per room.
# # Data Preparation for Modeling.

# In[40]:


# Separating the independent variables (X) and the dependent variable (Y)
X = data.drop(["booking_status"], axis=1)
Y = data["booking_status"]

X = pd.get_dummies(X, drop_first=True) # Encoding the Categorical features


# In[41]:


# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,stratify=Y, random_state=1)


# In[42]:


print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# In[43]:


# Creating metric function 
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))

    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Cancelled', 'Cancelled'], yticklabels=['Not Cancelled', 'Cancelled'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


Question 4.1: Build a Logistic Regression model
# In[44]:


lg = LogisticRegression()
lg.fit(X_train,y_train)


# In[45]:


# Checking the performance on the training data
y_pred_train = lg.predict(X_train)
metrics_score(y_train,y_pred_train)

Reading the confusion matrix (clockwise):
True Negative (Actual=0, Predicted=0): Model predicts that the booking would not be cancelled and the booking was not cancelled.

False Positive (Actual=0, Predicted=1): Model predicts that the booking would be cancelled but the booking was not cancelled.

False Negative (Actual=1, Predicted=0): Model predicts that the booking would not be cancelled and the booking was cancelled.

True Positive (Actual=1, Predicted=1): Model predicts that he booking would be cancelled and the booking was cancelled.

# In[46]:


# Checking the performance on the test dataset
y_pred_train = lg.predict(X_test)
metrics_score(y_test,y_pred_train)

Question 4.3:
Find the optimal threshold for the model using the Precision-Recall Curve
# In[47]:


# Predict_proba gives the probability of each observation belonging to each class
y_scores_lg=lg.predict_proba(X_train)

precisions_lg, recalls_lg, thresholds_lg = precision_recall_curve(y_train, y_scores_lg[:,1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_lg, precisions_lg[:-1], 'b--', label='precision')
plt.plot(thresholds_lg, recalls_lg[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()

We can see that precision and recall are balanced for a threshold of about ~0.45
# In[48]:


# Setting the optimal threshold
optimal_threshold = 0.45

Question 4.4:
Check the performance of the model on train and test data using the optimal threshold.
# In[49]:


# Creating confusion matrix
# Checking the model performance on train data
y_pred_train = lg.predict_proba(X_train)
metrics_score(y_train, y_pred_train[:,1]>optimal_threshold)

The model performance has improved. The recall has increased significantly for class 1
# In[50]:


# Checking the model performance on test data
y_pred_test = lg.predict_proba(X_test)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold)

The recall of the test data has increased significantly while at the same time, the precision has decreased, which is to be expected while adjusting the threshold.Question 5: Support Vector Machines
# In[57]:


# To accelerate SVM training, let's scale the data for support vector machines.
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train_scaled = scaling.transform(X_train)
X_test_scaled = scaling.transform(X_test)

Question 5.1: Build a Support Vector Machine model using a linear kernel.
# In[58]:


# using the scaled data for modeling Support Vector Machine

svm = SVC(kernel='linear',probability=True) # Linear kernal or linear decision boundary
model = svm.fit(X= X_train_scaled, y = y_train)      

Question 5.2: Check the performance of the model on train and test data
# In[59]:


# Checking model performance on train set
y_pred_train_svm = model.predict(X_train_scaled)

metrics_score(y_train, y_pred_train_svm)


# In[60]:


#Checking model performance on test set4
y_pred_test_svm = model.predict(X_test_scaled)
metrics_score(y_test, y_pred_test_svm)

SVM model with linear kernel is not overfitting as the accuracy is around 80% for both train and test dataset

The Recall for the model is around 61% implying that our model will not correctly predict the bookings that would be cancelledQuestion 5.3: Find the optimal threshold for the model using the Precision-Recall Curve
# In[61]:


# Predict on train data
y_scores_svm=model.predict_proba(X_train_scaled)

precisions_svm, recalls_svm, thresholds_svm =  precision_recall_curve(y_train, y_scores_svm[:,1])


# In[62]:


# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_svm, precisions_svm[:-1], 'b--', label='precision')
plt.plot(thresholds_svm, recalls_svm[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()


# In[63]:


optimal_threshold_svm= 0.4

Question 5.4: Check the performance of the model on train and test data using the optimal threshold.
# In[64]:


# Check the performance of the model on train data
y_pred_train_svm = model.predict_proba(X_train_scaled)
metrics_score(y_train, y_pred_train[:,1]>optimal_threshold)


# In[65]:


# Check the performance of the model on test data
y_pred_train = model.predict_proba(X_test_scaled)
metrics_score(y_test, y_pred_train[:,1]>optimal_threshold)


# In[66]:


y_pred_test = model.predict_proba(X_test_scaled)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold)


# # Question 5.5: Build a Decision Tree Model

# In[74]:


model_dt = DecisionTreeClassifier(class_weight = {0: 0.17, 1: 0.83}, random_state = 1)
model_dt.fit(X_train, y_train)

Question 5.6: Check the performance of the model on train and test data
# In[75]:


# Checking performance on the training dataset
pred_train_dt = model_dt.predict(X_train)
metrics_score(y_train, pred_train_dt)


# In[76]:


# Checking performance on the test dataset
pred_test_dt = model_dt.predict(X_test)
metrics_score(y_test, pred_test_dt)


# # Visualizing the Decision Tree

# In[81]:


feature_names = list(X_train.columns)
plt.figure(figsize=(20, 10))
out = tree.plot_tree(
    estimator,max_depth=3,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
# below code will add arrows to the decision tree split if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()

Question 6.5: What are some important features based on the tuned decision tree?
# In[82]:


# Importance of features in the tree building

importances = estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

According to this model, Lead time, Market segment Type and Number of Special request are the 3 most important features that describe why a guest would cancel his/her booking.


After tuning the model, we found out that only 3 features are important. It seems like the model is having high bias, as it has over-simplified the problem and is not capturing the patterns associated with other variables.

# # Question 7.1: Build a Random Forest Model

# In[83]:


# Fitting the Random Forest classifier on the training data
rf_estimator = RandomForestClassifier(class_weight = {0: 0.17, 1: 0.83}, random_state = 1)

rf_estimator.fit(X_train, y_train)

Question 7.2: Check the performance of the model on the train and test data.
# In[84]:


# Checking performance on the training data
y_pred_train_rf = rf_estimator.predict(X_train)

metrics_score(y_train, y_pred_train_rf)


# In[85]:


# Checking performance on the testing data
y_pred_test_rf = rf_estimator.predict(X_test)

metrics_score(y_test, y_pred_test_rf)

Question 7.3: What are some important features based on the Random Forest?
# In[86]:


importances = rf_estimator.feature_importances_

columns = X.columns

importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)

plt.figure(figsize = (13, 13))

sns.barplot(importance_df.Importance, importance_df.index)

The Random Forest further verifies the results from the decision tree that the most important features are Lead Time, Average Price per Room and Number of Special Request.Question 8: Conclude ANY FOUR key takeaways for business recommendations.1.The hotel should establish an online presence and should keep an eye on social review.

2.The hotel should be as quick as it can in handling customers' requirements and providing them with the best assistance.

3.The hotel should take advantage of peak booking seasons.

4.The hotel should train its staff to deliver world-class service and give customers an innovative experience all the time.

# In[ ]:




