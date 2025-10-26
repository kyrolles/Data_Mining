# What is the data about?
    # the data is about animes aird until 2023 

import pandas as pd

# Load the dataset
file_path = '/home/kyrl/dataMining/anime-dataset-2023.csv'
anime_data = pd.read_csv(file_path)

print("""
============================
HOW MANY ATTRIBUTES DESCRIBE THE DATA? WHAT ARE THE TYPES OF THESE ATTRIBUTES?
============================
""")
# Display the first few rows and column information
print(anime_data.head())
print(anime_data.info())
print("""
============================
Are there missing values? If yes, propose a method to deal with missing values.
============================
""")
# Check for missing values
missing_values = anime_data.isnull().sum()

# Display columns with missing values
print("Missing values in each column:")
print(missing_values)

# Convert the 'Score' column to numeric, coercing invalid values to NaN
anime_data['Score'] = pd.to_numeric(anime_data['Score'], errors='coerce')

# Fill missing values in the 'Score' column with the mean
anime_data['Score'].fillna(anime_data['Score'].mean(), inplace=True)

print("""
============================
If the data has numeric attributes, choose at least two attributes and define their
distributions. Represent these distributions using BoxPlots. Which kind of conclusions you
derive from these representations? Are there any outliers?
(to draw boxpolt you can use the online demonstration provided in
http://onlinestatbook.com/chapter2/boxplot_demo.html. Clique on show demonstration to
see an example. Then, if you copy paste the values of your distribution, the relevant boxplot
will be shown. )
============================
""")

# Convert 'Score' and 'Episodes' to numeric, coercing invalid values to NaN
anime_data['Score'] = pd.to_numeric(anime_data['Score'], errors='coerce')
anime_data['Episodes'] = pd.to_numeric(anime_data['Episodes'], errors='coerce')

# Drop rows with missing values in 'Score' or 'Episodes'
anime_data.dropna(subset=['Score', 'Episodes'], inplace=True)

# Calculate basic statistics for 'Score' and 'Episodes'
score_stats = anime_data['Score'].describe()
episodes_stats = anime_data['Episodes'].describe()

print("Score Statistics:")
print(score_stats)

print("\nEpisodes Statistics:")
print(episodes_stats)

# Export the values to a CSV for easy copy-pasting
# anime_data[['Score', 'Episodes']].to_csv('numeric_attributes.csv', index=False)


print("""
============================
Define  how  to  measure  the  similarity  between  the  data  objects  according  to  the  attribute
types of your datasets?
============================
""")
# For numeric attributes, we can use Euclidean distance to measure similarity.
import numpy as np
# Example data for two anime objects
anime1 = anime_data.loc[0, ['Score', 'Episodes']].values
anime2 = anime_data.loc[1, ['Score', 'Episodes']].values
# Calculate Euclidean Distance
euclidean_distance = np.sqrt(np.sum((anime1 - anime2) ** 2))
print(f"Euclidean Distance: {euclidean_distance}")



# For categorical attributes, we can use Jaccard Similarity to measure similarity.
# Example data for two anime objects
genres1 = set(anime_data.loc[0, 'Genres'].split(', '))
genres2 = set(anime_data.loc[1, 'Genres'].split(', '))
# Calculate Jaccard Similarity
jaccard_similarity = len(genres1.intersection(genres2)) / len(genres1.union(genres2))
print(f"Jaccard Similarity: {jaccard_similarity}")



# For string attributes, we can use Levenshtein Distance to measure similarity.
import Levenshtein
# Example data for two anime objects
name1 = anime_data.loc[0, 'Name']
name2 = anime_data.loc[1, 'Name']
# Calculate Levenshtein Distance
levenshtein_distance = Levenshtein.distance(name1, name2)
print(f"Levenshtein Distance: {levenshtein_distance}")

print("""
============================
Define the data mining tasks that can be performed on the chosen datasets. 
============================
""")
# Data mining tasks that can be performed on the anime dataset include:
# 1. Classification: Predicting the genre of an anime based on its attributes (e.g., score, episodes).
#    This can be done using algorithms like Decision Trees, Random Forests, or Support Vector Machines.
# 2. Clustering: Grouping similar animes together based on their attributes (e.g., score, episodes, genres).
#    This can be done using algorithms like K-Means or Hierarchical Clustering.
# 3. Association Rule Mining: Finding interesting relationships between different genres and scores.
#    This can be done using algorithms like Apriori or FP-Growth.
# 4. Regression: Predicting the score of an anime based on its attributes (e.g., episodes, genres).
#    This can be done using algorithms like Linear Regression or Decision Trees.
# 5. Recommendation Systems: Recommending animes to users based on their preferences and viewing history.
#    This can be done using collaborative filtering or content-based filtering techniques.



print("""
============================
Step 3 & 4
============================
""")    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the dataset
file_path = '/home/kyrl/dataMining/anime-dataset-2023.csv'
anime_data = pd.read_csv(file_path)

# Handle missing values
anime_data['Score'] = pd.to_numeric(anime_data['Score'], errors='coerce')
anime_data['Episodes'] = pd.to_numeric(anime_data['Episodes'], errors='coerce')
anime_data['Popularity'] = pd.to_numeric(anime_data['Popularity'], errors='coerce')
anime_data['Rank'] = pd.to_numeric(anime_data['Rank'], errors='coerce')

# Fill missing values with median
anime_data['Score'].fillna(anime_data['Score'].median(), inplace=True)
anime_data['Episodes'].fillna(anime_data['Episodes'].median(), inplace=True)
anime_data['Popularity'].fillna(anime_data['Popularity'].median(), inplace=True)
anime_data['Rank'].fillna(anime_data['Rank'].median(), inplace=True)

# Filter dataset to keep only rows with Type as TV, Movie, or OVA (for simplicity)
anime_data = anime_data[anime_data['Type'].isin(['TV', 'Movie', 'OVA'])]

# Select relevant features
features = ['Score', 'Episodes', 'Popularity', 'Rank']
X = anime_data[features]
y = anime_data['Type']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)





# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")




# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate performance
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, target_names=label_encoder.classes_))

# # Visualize the decision tree
# plt.figure(figsize=(15, 10))
# plot_tree(dt_model, feature_names=features, class_names=label_encoder.classes_, filled=True)
# plt.title("Decision Tree for Anime Type Classification")
# plt.show()


# Create and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_nb = nb_model.predict(X_test_scaled)

# Evaluate performance
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb, target_names=label_encoder.classes_))





# Create and train the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate performance
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))




# Create and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_knn = knn_model.predict(X_test_scaled)

# Evaluate performance
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn, target_names=label_encoder.classes_))



# Compare accuracies
models = ['Decision Tree', 'Naive Bayes', 'Logistic Regression', 'KNN']
accuracies = [accuracy_dt, accuracy_nb, accuracy_lr, accuracy_knn]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.title('Classification Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
plt.show()

# Create confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

models_preds = [y_pred_dt, y_pred_nb, y_pred_lr, y_pred_knn]
model_names = models

for i, (preds, name) in enumerate(zip(models_preds, model_names)):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    axes[i].set_title(f'Confusion Matrix: {name}')
    axes[i].set_ylabel('Actual')
    axes[i].set_xlabel('Predicted')

plt.tight_layout()
plt.show()