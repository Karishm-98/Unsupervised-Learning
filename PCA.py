import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
# Plot the covariance matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


cancer = load_breast_cancer(as_frame=True)
df=pd.DataFrame(cancer.data,columns=cancer.feature_names)
df['target']=cancer.target
print(df)
X=df[cancer.feature_names]
print(X)
scaler=StandardScaler()
X_std=np.round(scaler.fit_transform(X),6)
print(X_std.shape)
cov_matrix=np.cov(X_std,rowvar=False)
print(cov_matrix.shape)
sns.heatmap(cov_matrix)
eigen_val,eigen_vec=np.linalg.eig(cov_matrix)
print(eigen_val)
idx=np.argsort(eigen_val)[::-1] 
eigen_val=eigen_val[idx]
eigen_vec=eigen_vec[:,idx]
explained_var=eigen_val/np.sum(eigen_val)
cumu_var=np.cumsum(explained_var)
threshold=0.95
k = np.argmax(cumu_var >= threshold) + 1
print(k)
u=eigen_vec[:,:k]
print(u.shape)
Z_pca=np.dot(X_std,u)
print("--------------------")
print(Z_pca)
print(Z_pca.shape)
df_z_pca=pd.DataFrame(Z_pca,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
print(df_z_pca)
print()
print("************************************************************")
print()
print("PCA using Sklearn")
# Let's say, components = 2
pca = PCA(n_components=10)
pca.fit(X_std)
x_pca = pca.transform(X_std)

# Create the dataframe with dynamic column names based on n_components
df_pca1 = pd.DataFrame(x_pca,
                       columns=['PC{}'.format(i+1) for i in range(pca.n_components_)])

# Print the resulting dataframe
print(df_pca1)




