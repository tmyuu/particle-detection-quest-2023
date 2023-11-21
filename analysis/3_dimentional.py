import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss

# Read the pickle file
df = pd.read_pickle("../work/input/LSWMD_25519.pkl")

# Step 1: Extract the numeric part of 'lotName'
df['lotNumber'] = df['lotName'].str.extract('(\d+)').astype(float)

# Step 2: Apply PCA to reduce the dimensionality
# Combine 'dieSize', 'lotNumber', and 'waferIndex' into a single DataFrame
X = df[['dieSize', 'lotNumber', 'waferIndex']]

# Standardize the features before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 1 dimension
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# Step 3: Analyze correlation with 'failureType'
# Since 'failureType' is categorical, traditional correlation won't work
# One way to approach this is to encode 'failureType' numerically and then use a correlation measure
df['failureType_encoded'] = df['failureType'].astype('category').cat.codes
correlation, p_value = ss.pearsonr(X_pca.flatten(), df['failureType_encoded'])

print(f"Correlation: {correlation}, P-value: {p_value}")
