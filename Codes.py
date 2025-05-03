#!/usr/bin/env python
# coding: utf-8

# # "A Machine Learning Approach to Classifying and Clustering Global Cities Based on Socioeconomic Indicators"

# # 1️⃣ Veri Ön İşleme

# In[52]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Eğer dosya noktalı virgül ile ayrılmışsa (Avrupa'da yaygın)
df = pd.read_csv("Data.csv", encoding='ISO-8859-1', sep=';')

print(df[features].dtypes)



# Sayısal sütunlar
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']


# Normalizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])


# In[53]:


df


# # Correlation

# In[54]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Numeric feature columns
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']

# Load the dataset (using ISO-8859-1 encoding and semicolon separator)
df = pd.read_csv("Data.csv", encoding='ISO-8859-1', sep=';')

# Convert comma to dot for decimal values and strip whitespaces
df[features] = df[features].applymap(lambda x: str(x).replace(",", ".").strip() if isinstance(x, str) else x)

# Convert to numeric types, set non-convertible values to NaN
df[features] = df[features].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values in the selected features
df = df.dropna(subset=features)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# === 1. Numerical Correlation Matrix ===
corr_matrix = df[features].corr()
print("\nCorrelation Matrix (Numerical):\n")
print(corr_matrix)

# === 2. Visual Correlation Heatmap ===
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


# # K-means Clustering

# In[109]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Normalizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Elbow method to find the optimal number of clusters
inertia = []
silhouette_scores = []

for k in range(2, 16):  # Increased range up to 15
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    
    # Elbow method (Inertia)
    inertia.append(kmeans.inertia_)
    
    # Silhouette score
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# Choose the optimal k (based on both Elbow and Silhouette method)
optimal_k = 5  # Update this based on your visual analysis

# Plotting the Elbow method and Silhouette score in a single column with two rows
plt.figure(figsize=(10, 12))

# Elbow method plot (First row)
plt.subplot(2, 1, 1)
plt.plot(range(2, 16), inertia, marker='o', label='Inertia')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.legend()

# Silhouette score plot (Second row)
plt.subplot(2, 1, 2)
plt.plot(range(2, 16), silhouette_scores, marker='o', color='orange', label='Silhouette Score')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.title("Silhouette Score for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.legend()

plt.tight_layout()
plt.show()

# Apply KMeans with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


# #  1. Import Libraries and Prepare Data for K-Means

# In[110]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleyelim
df = pd.read_csv("Data.csv", encoding='ISO-8859-1', sep=';')

# Kullanılacak sayısal sütunlar
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']

# Veriyi normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])


# # 2. Apply KMeans Clustering with k=5

# In[111]:


# Create and fit the KMeans model with 4 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


# # 3. Cluster-wise Mean Analysis

# In[112]:


# Calculate the average value of each feature within each cluster
cluster_summary = df.groupby('Cluster')[features].mean()
print("Cluster Feature Averages:")
print(cluster_summary)


# # 4. Visualize Cluster Sizes

# In[113]:


# Count how many cities belong to each cluster
cluster_counts = df['Cluster'].value_counts().sort_index()
print("\nNumber of Cities per Cluster:")
print(cluster_counts)

# Visual bar plot of cluster distribution
sns.countplot(x='Cluster', data=df)
plt.title("Number of Cities in Each Cluster (k=4)")
plt.xlabel("Cluster")
plt.ylabel("City Count")
plt.show()


# # Kümelerin Ne Kadar İyi Ayrıldığını Testler İle sınayalım 

# # 1. Davies-Bouldin Index (DBI)
# Küme içi benzerlik ve kümeler arası farkı ölçer. Düşük değerler daha iyidir.

# In[114]:


from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(X_scaled, df['Cluster'])
print(f"Davies-Bouldin Index (k=6): {db_score:.4f}")


# # ✅ 2. Calinski-Harabasz Index (Variance Ratio Criterion)
# Yüksek değerler daha iyi küme ayrımını ifade eder.

# In[115]:


from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X_scaled, df['Cluster'])
print(f"Calinski-Harabasz Index (k=6): {ch_score:.4f}")


# # ✅ 3. Visual Evaluation with PCA or t-SNE
# Kümelerin görsel ayrımı için önemli.

# In[116]:


from sklearn.decomposition import PCA
import seaborn as sns

# PCA ile 2 boyuta indirgeme
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title("PCA ile Görselleştirilmiş Kümeler (k=6)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")
plt.show()


# # 🔁 4. Tüm metrikler için farklı k değerlerini karşılaştır
# Bu sayede k=6 gerçekten optimum mu kontrol edebiliriz.

# In[117]:


db_scores = []
ch_scores = []

for k in range(2, 16):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))

# Sonuçları görselleştir
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 16), db_scores, marker='o')
plt.axvline(x=optimal_k, color='red', linestyle='--')
plt.title("Davies-Bouldin Index")
plt.xlabel("Number of Clusters")
plt.ylabel("DB Index (Lower is Better)")

plt.subplot(1, 2, 2)
plt.plot(range(2, 16), ch_scores, marker='o', color='green')
plt.axvline(x=optimal_k, color='red', linestyle='--')
plt.title("Calinski-Harabasz Index")
plt.xlabel("Number of Clusters")
plt.ylabel("CH Score (Higher is Better)")

plt.tight_layout()
plt.show()


# In[120]:


from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv("Data.csv", encoding='ISO-8859-1', sep=';')

# Features to be used
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Calculate Davies-Bouldin and Calinski-Harabasz scores for different k values
db_scores = []
ch_scores = []
k_values = range(2, 16)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))

# Plot the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, db_scores, marker='o')
plt.axvline(x=5, color='red', linestyle='--')
plt.title("Davies-Bouldin Index")
plt.xlabel("Number of Clusters")
plt.ylabel("DB Index (Lower is Better)")

plt.subplot(1, 2, 2)
plt.plot(k_values, ch_scores, marker='o', color='green')
plt.axvline(x=5, color='red', linestyle='--')
plt.title("Calinski-Harabasz Index")
plt.xlabel("Number of Clusters")
plt.ylabel("CH Score (Higher is Better)")

plt.tight_layout()
plt.show()

# Return numerical results
list(zip(k_values, db_scores, ch_scores))


# # 📌 1. KMeans ile k=5 Kümeleme

# In[122]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Veriyi yükle
df = pd.read_csv("Data.csv", encoding='ISO-8859-1', sep=';')

# Özellikler
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']

# Normalizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# KMeans ile k=5 kümeleme
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


# # 📊 2. Her Kümenin Ortalama Skorları

# In[123]:


cluster_summary = df.groupby('Cluster')[features].mean().round(2)
print(cluster_summary)


# # 3. Her Kümeden Örnek Şehirler

# In[124]:


sample_cities = df.groupby('Cluster').apply(lambda x: x[['City', 'Country']].head(3)).reset_index(drop=True)
print(sample_cities)


# In[125]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA ile 2 boyuta indirgeme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Görselleştirme
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='tab10', alpha=0.6)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KMeans Clustering (k=5) - PCA ile 2D Görselleştirme")
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()


# In[126]:


import numpy as np

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='tab10', alpha=0.6)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KMeans Clustering (k=5) - PCA ile Etiketli Görselleştirme")
plt.colorbar(scatter, label='Cluster')
plt.grid(True)

# Her kümeden 5 şehir etiketle
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    sample = cluster_data.sample(5, random_state=42)
    for i, row in sample.iterrows():
        plt.text(X_pca[i, 0], X_pca[i, 1], row['City'], fontsize=8)

plt.tight_layout()
plt.show()


# In[128]:


plt.figure(figsize=(16, 12))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='tab10', alpha=0.7)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KMeans Clustering (k=5) - PCA ile Etiketli Tüm Şehirler")
plt.colorbar(scatter, label='Cluster')
plt.grid(True)

# Tüm şehirleri etiketle
for i, row in df.iterrows():
    plt.text(X_pca[i, 0], X_pca[i, 1], row['City'], fontsize=6, alpha=0.7)

plt.tight_layout()
plt.show()


# In[131]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("Data.csv", encoding='ISO-8859-1', sep=';')

# Features for clustering
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Apply KMeans with k=5
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA transformation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot with all city labels
plt.figure(figsize=(16, 12))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='tab10', alpha=0.6)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KMeans Clustering (k=5) - PCA ile Etiketli Tüm Şehirler")
plt.colorbar(scatter, label='Cluster')
plt.grid(True)

# Add labels
for i, row in df.iterrows():
    plt.text(X_pca[i, 0], X_pca[i, 1], row['City'], fontsize=5, alpha=0.7)

# Save the plot as PNG
output_path = "kmeans_pca_all_labels.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

output_path


# In[141]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("Data.csv", encoding='ISO-8859-1', sep=';')

# Features for clustering
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Apply KMeans with k=5
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA transformation for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Plot with cluster color and legend
plt.figure(figsize=(16, 12))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for cluster in range(5):
    clustered_data = df[df['Cluster'] == cluster]
    plt.scatter(clustered_data['PCA1'], clustered_data['PCA2'],
                label=f'Cluster {cluster}', alpha=0.6, s=60, c=colors[cluster])

# City labels
for i, row in df.iterrows():
    plt.text(row['PCA1'], row['PCA2'], row['City'],
             fontsize=5, alpha=0.7, color='black', fontweight='black')

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KMeans Clustering (k=5) - PCA Reduction with Cluster Legend")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()

# Save the plot
output_path = "kmeans_pca_with_legend.png"
plt.savefig(output_path, dpi=300)
plt.close()


# # 5. Evaluate Cluster Profiles Using Radar Chart and Machine Learning Classifiers

# # 3️⃣ Küme Başarı Analizi
# Açıklama:
# Oluşturulan kümelerin ne kadar anlamlı olduğunu farklı sınıflandırma algoritmalarıyla test edeceğiz. Bu, kümelerin ayırt edilebilirliğini doğrulamak açısından önemlidir.
# 
# Kullanılan Algoritmalar:
# Logistic Regression
# 
# SVC (Support Vector Machine)
# 
# Random Forest
# 
# Python Kodu:

# In[142]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Küme etiketlerini tahmin etmeye çalışıyoruz
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Cluster'], test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

# SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM:\n", classification_report(y_test, y_pred_svm))

# Lojistik Regresyon
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))


# In[145]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Features and target
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']
X = df[features]
y = df['Cluster']  # Cluster labels assigned earlier

# Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
rf = RandomForestClassifier(random_state=42)
svm = SVC()
lr = LogisticRegression(max_iter=1000)

# Fit and Predict
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
lr.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_lr = lr.predict(X_test)

# Confusion Matrices
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Visualization
plt.figure(figsize=(10, 15))

# Random Forest
plt.subplot(3, 1, 1)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# SVM
plt.subplot(3, 1, 2)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Logistic Regression
plt.subplot(3, 1, 3)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# Classification Reports
print("Random Forest:\n", classification_report(y_test, y_pred_rf))
print("SVM:\n", classification_report(y_test, y_pred_svm))
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))



# Confusion Matrices
print("Random Forest:\n",cm_rf)
print("SVM:\n",cm_svm)
print("Logistic Regression:\n",cm_lr)


# In[97]:


print(df.head())  # İlk birkaç satırı inceleyin


# # Enhancing Machine Learning Accuracy Through Grid Search and Cross-Validation

# In[147]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np

# Özellikler ve hedef
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']
X = df[features]
y = df['Cluster']

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold objesi
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# RANDOM FOREST
param_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
gs_rf = GridSearchCV(rf, param_rf, cv=5, scoring='accuracy', n_jobs=-1)
gs_rf.fit(X_scaled, y)
print("Random Forest Best Params:", gs_rf.best_params_)
print("Random Forest CV Score:", gs_rf.best_score_)
cv_score_rf = cross_val_score(gs_rf.best_estimator_, X_scaled, y, cv=kfold)
print("Random Forest Mean CV Accuracy:", np.mean(cv_score_rf))


# SVM
param_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm = SVC()
gs_svm = GridSearchCV(svm, param_svm, cv=5, scoring='accuracy', n_jobs=-1)
gs_svm.fit(X_scaled, y)
print("SVM Best Params:", gs_svm.best_params_)
print("SVM CV Score:", gs_svm.best_score_)
cv_score_svm = cross_val_score(gs_svm.best_estimator_, X_scaled, y, cv=kfold)
print("SVM Mean CV Accuracy:", np.mean(cv_score_svm))


# LOGISTIC REGRESSION
param_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
lr = LogisticRegression(max_iter=1000)
gs_lr = GridSearchCV(lr, param_lr, cv=5, scoring='accuracy', n_jobs=-1)
gs_lr.fit(X_scaled, y)
print("Logistic Regression Best Params:", gs_lr.best_params_)
print("Logistic Regression CV Score:", gs_lr.best_score_)
cv_score_lr = cross_val_score(gs_lr.best_estimator_, X_scaled, y, cv=kfold)
print("Logistic Regression Mean CV Accuracy:", np.mean(cv_score_lr))


# In[152]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

# Özellikler ve hedef
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']
X = df[features]
y = df['Cluster']

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold objesi
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Performans sonuçlarını saklamak için liste
results = []

# ------------------- RANDOM FOREST -------------------
param_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
gs_rf = GridSearchCV(rf, param_rf, cv=5, scoring='accuracy', n_jobs=-1)
gs_rf.fit(X_scaled, y)
cv_score_rf = cross_val_score(gs_rf.best_estimator_, X_scaled, y, cv=kfold)
mean_score_rf = np.mean(cv_score_rf)
print("Random Forest Best Params:", gs_rf.best_params_)
print("Random Forest Mean CV Accuracy:", mean_score_rf)
results.append(("Random Forest", mean_score_rf))

# ------------------- SVM -------------------
param_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm = SVC()
gs_svm = GridSearchCV(svm, param_svm, cv=5, scoring='accuracy', n_jobs=-1)
gs_svm.fit(X_scaled, y)
cv_score_svm = cross_val_score(gs_svm.best_estimator_, X_scaled, y, cv=kfold)
mean_score_svm = np.mean(cv_score_svm)
print("SVM Best Params:", gs_svm.best_params_)
print("SVM Mean CV Accuracy:", mean_score_svm)
results.append(("SVM", mean_score_svm))

# ------------------- LOGISTIC REGRESSION -------------------
param_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
lr = LogisticRegression(max_iter=1000)
gs_lr = GridSearchCV(lr, param_lr, cv=5, scoring='accuracy', n_jobs=-1)
gs_lr.fit(X_scaled, y)
cv_score_lr = cross_val_score(gs_lr.best_estimator_, X_scaled, y, cv=kfold)
mean_score_lr = np.mean(cv_score_lr)
print("Logistic Regression Best Params:", gs_lr.best_params_)
print("Logistic Regression Mean CV Accuracy:", mean_score_lr)
results.append(("Logistic Regression", mean_score_lr))

# ------------------- GÖRSELLEŞTİRME -------------------
# Veriyi dataframe'e çevir
results_df = pd.DataFrame(results, columns=["Model", "Mean CV Accuracy"])

# Bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Mean CV Accuracy", data=results_df, palette="viridis")
plt.title("Model Comparison (Mean CV Accuracy)", fontsize=14)
plt.ylabel("Mean Accuracy")
plt.ylim(0, 1)
for index, row in results_df.iterrows():
    plt.text(index, row["Mean CV Accuracy"] + 0.01, f"{row['Mean CV Accuracy']:.2f}", ha='center')
plt.tight_layout()
plt.show()


# In[149]:


plt.figure(figsize=(8, 5))
sns.barplot(x="Mean CV Accuracy", y="Model", data=results_df, palette="coolwarm")
plt.title("Model Comparison (Mean CV Accuracy)", fontsize=14)
plt.xlabel("Mean Accuracy")
plt.xlim(0, 1)
for index, row in results_df.iterrows():
    plt.text(row["Mean CV Accuracy"] + 0.01, index, f"{row['Mean CV Accuracy']:.2f}", va='center')
plt.tight_layout()
plt.show()


# In[ ]:





# # Visualization of Feature Importance

# In[ ]:





# In[ ]:





# In[174]:


import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Calculate Feature Importance with Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_scaled, y)

# Get feature importances
importances_rf = rf_model.feature_importances_

# Calculate Feature Importance with XGBoost Model
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_scaled, y)

# Get feature importances
importances_xgb = xgb_model.feature_importances_

# Get feature names
features = X.columns

# Sort the feature importances for comparison
indices_rf = np.argsort(importances_rf)
indices_xgb = np.argsort(importances_xgb)

# Plot the feature importance for Random Forest
plt.figure(figsize=(12, 6))

# Random Forest subplot
plt.subplot(1, 2, 1)
plt.title("Feature Importance (Random Forest)")
bars_rf = plt.barh(range(len(indices_rf)), importances_rf[indices_rf], align="center")
plt.yticks(range(len(indices_rf)), [features[i] for i in indices_rf])
plt.xlabel("Feature Importance")

# Adding numerical labels on Random Forest bars
for bar in bars_rf:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', va='center')

# XGBoost subplot
plt.subplot(1, 2, 2)
plt.title("Feature Importance (XGBoost)")
bars_xgb = plt.barh(range(len(indices_xgb)), importances_xgb[indices_xgb], align="center")
plt.yticks(range(len(indices_xgb)), [features[i] for i in indices_xgb])
plt.xlabel("Feature Importance")

# Adding numerical labels on XGBoost bars
for bar in bars_xgb:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', va='center')

plt.tight_layout()
plt.show()

# Print numerical feature importance values
print("Random Forest Feature Importance:")
for feature, importance in zip(features, importances_rf):
    print(f"{feature}: {importance:.4f}")

print("\nXGBoost Feature Importance:")
for feature, importance in zip(features, importances_xgb):
    print(f"{feature}: {importance:.4f}")


# # PCA

# In[180]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print(f"Explained Variance Ratio (PC1): {pca.explained_variance_ratio_[0]:.2f}")
print(f"Explained Variance Ratio (PC2): {pca.explained_variance_ratio_[1]:.2f}")

# Loadings for PC1 and PC2
pca_components = pca.components_

print("\nPrincipal Component 1 Loadings:")
for feature, loading in zip(features, pca_components[0]):
    print(f"{feature}: {loading:.4f}")

print("\nPrincipal Component 2 Loadings:")
for feature, loading in zip(features, pca_components[1]):
    print(f"{feature}: {loading:.4f}")

# Visualize PCA biplot (loadings + scatter)
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
for i, feature in enumerate(features):
    plt.arrow(0, 0, pca_components[0, i]*3, pca_components[1, i]*3,
              color='red', alpha=0.7, head_width=0.05)
    plt.text(pca_components[0, i]*3.2, pca_components[1, i]*3.2,
             f"{feature}\n({pca_components[0, i]:.2f}, {pca_components[1, i]:.2f})",
             color='black', ha='center', va='center')

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("PCA Biplot with Feature Loadings")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[181]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create PCA model and use it to reduce dimensionality
pca = PCA(n_components=2)  # Taking the first two components
X_pca = pca.fit_transform(X_scaled)

# Print explained variance ratio of the components
print(f"Explained Variance Ratio (PC1): {pca.explained_variance_ratio_[0]:.2f}")
print(f"Explained Variance Ratio (PC2): {pca.explained_variance_ratio_[1]:.2f}")

# Visualize the data in two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title("Dimensionality Reduction with PCA", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Target Variable')
plt.tight_layout()
plt.show()
# Get the components (loadings) for PC1 and PC2
pca_components = pca.components_

# Display the loadings for the first two principal components
print("Principal Component 1 Loadings:")
for feature, loading in zip(features, pca_components[0]):
    print(f"{feature}: {loading:.4f}")

print("\nPrincipal Component 2 Loadings:")
for feature, loading in zip(features, pca_components[1]):
    print(f"{feature}: {loading:.4f}")



# In[182]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Dummy feature names and simulated X_scaled for illustration
features = ['Economics', 'Human Capital', 'Quality of Life', 'Environment', 'Governance']
np.random.seed(42)
X_scaled = np.random.rand(100, 5)
y = np.random.randint(0, 3, size=100)

# PCA analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_components = pca.components_

# Create a dataframe for loadings
loadings_df = pd.DataFrame(pca_components.T, columns=['PC1', 'PC2'], index=features)

# Plot the loadings
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# PC1 Loadings
ax[0].barh(loadings_df.index, loadings_df['PC1'], color='skyblue')
for i, v in enumerate(loadings_df['PC1']):
    ax[0].text(v + 0.01 if v >= 0 else v - 0.15, i, f"{v:.2f}", va='center')
ax[0].set_title("Principal Component 1 Loadings")
ax[0].set_xlabel("Loading Value")

# PC2 Loadings
ax[1].barh(loadings_df.index, loadings_df['PC2'], color='lightgreen')
for i, v in enumerate(loadings_df['PC2']):
    ax[1].text(v + 0.01 if v >= 0 else v - 0.15, i, f"{v:.2f}", va='center')
ax[1].set_title("Principal Component 2 Loadings")
ax[1].set_xlabel("Loading Value")

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




