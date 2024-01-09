# Part 1
# October 7, 2023
# by Olivia Radcliiffe

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------- Task 1- Data Preparation: (20%) -----------------------------
print("\nTask 1- Data Preparation:")

# ---- 1. Show the distribution of data in two classes (e.g., using a barplot) in the combined dataset ----
print("Task 1 - 1")

# importing the dataset
directory = os.path.dirname(os.path.abspath(sys.argv[0]))
labels_df = pd.read_csv(directory + '/data/actual.csv')

classes = list(['ALL (Acute Lymphoblastic Leukemia)', 'AML (Acute Myeloid Leukemia)'])
counts = list([(labels_df['cancer'] == 'ALL').sum(), (labels_df['cancer'] == 'AML').sum()])
  
fig = plt.figure()
print("See figure")
 
# creating the bar plot
plt.bar(classes, counts, color ='red')
 
plt.xlabel("Cancer Type")
plt.ylabel("No. of patients")
plt.title("Distibution of Data in Classes")
plt.show()


# ---- 2. Encode the labels ----
print("Task 1 - 2")

# extract classes
values = np.array(labels_df['cancer'].ravel())

# endode classes to integers
labelEncoder = LabelEncoder()
integer_labels = labelEncoder.fit_transform(values)

print("Encoded labels: ", integer_labels)


# ---- 3. Remove all the ‘Call’ columns from both data files ----
print("Task 1 - 3")

# import train data file
train_data_df = pd.read_csv(directory + '/data/data_set_ALL_AML_train.csv')
cleaned_train_data_df = train_data_df.copy()

# remove ‘Call’ columns from train data file
for col in train_data_df.columns:
    if 'call' in col:
        cleaned_train_data_df = cleaned_train_data_df.drop(columns=col)

# import test data file
test_data_df = pd.read_csv(directory + '/data/data_set_ALL_AML_independent.csv')
cleaned_test_data_df = test_data_df.copy()

# remove ‘Call’ columns from test data file
for col in test_data_df.columns:
    if 'call' in col:
        cleaned_test_data_df = cleaned_test_data_df.drop(columns=col)

# data with removed 'Call' columns
print("Train data with 'Call' columns:\n", cleaned_train_data_df.head())
print("Test data with 'Call' columns:\n", cleaned_test_data_df.head())


# ---- 4. Associate the train and test data to the labels ----
print("Task 1 - 4")

# initialize  train and test data label lists
train_labels = []
test_labels = []

# find integer labels for train data and add to list
for patientNum in cleaned_train_data_df.columns[2:]:
    train_labels.append(integer_labels[int(patientNum)-1])

# find integer labels for test data and add to list
for col in cleaned_test_data_df.columns[2:]:
    test_labels.append(integer_labels[int(col)-1])

print("Train labels: ", train_labels)
print("Test labels: ", test_labels)


# ---- 5. Compute and display summary statistics for the data - Normalize the data, if necessary ----
print("Task 1 - 5:")

# standardize instance
scaler = StandardScaler()

# standardizing train data
standardized_train_data = scaler.fit_transform(cleaned_train_data_df.iloc[:,2:])
# Convert the standardized data back to a DataFrame     
standardized_train_df = pd.DataFrame(standardized_train_data, columns=cleaned_train_data_df.columns[2:])

tr_standardized_train_df = np.transpose(standardized_train_df)

# display train summary statistics
train_summary_stats = tr_standardized_train_df.describe()
print("Train summary stats: \n", train_summary_stats)

# standardizing test data
standardized_test_data = scaler.fit_transform(cleaned_test_data_df.iloc[:,2:])
# Convert the standardized data back to a DataFrame
standardized_test_df = pd.DataFrame(standardized_test_data, columns=cleaned_test_data_df.columns[2:])

tr_standardized_test_df = np.transpose(standardized_test_df)

# display test summary statistics
test_summary_stats = tr_standardized_test_df.describe()
print("Test summary stats: \n", test_summary_stats)



# -------------------------------- Task 2- Dimensionality Reduction: (20%) --------------------------
print("\nTask 2- Dimensionality Reduction:")

# ---- 1. Research and write a short paragraph on a high-level description of PCA method and how
#       it is used for reducing the length of the input feature vectors. ----
    
    # See PDF report


# ---- 2. Use PCA from sklearn to select features that account for 90% of data variance in trainset
print("Task 2 - 2")
      
# retain 90% of the variance 
pca = PCA(n_components=0.9)

# Fit the PCA model to the standardized training data
ninetyPCA = pca.fit_transform(tr_standardized_train_df)
print("Cumulative component variance: ",pca.explained_variance_ratio_.cumsum())

# Number of components/features selected
n_selected_features = pca.n_components_
print("Number of selected features: ", n_selected_features)

# Show the selected principal components
ninetyPCADf = pd.DataFrame(data = ninetyPCA)
print("Selected principal components (90%): \n", ninetyPCADf)


# ---- 3. Visualize the trainset in 3D space when the first 3 PCA components are selected ----

# use 3 PCA components
pca = PCA(n_components=3)
threeDpca = pca.fit_transform(tr_standardized_train_df)

# create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# extract the first three PCA components
x = threeDpca[:, 0]
y = threeDpca[:, 1]
z = threeDpca[:, 2]

colors = np.where(np.array(train_labels) == 0, 'b', 'r')

# create the scatter plot
scatter = ax.scatter(x, y, z, marker='o', c=colors, label='Labels')

# set labels and title
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('Train Data - PCA 3D Visualization')

# Create legend
legend_labels = ['ALL', 'AML']
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in ['b', 'r']]
ax.legend(legend_handles, legend_labels)

plt.show()