import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the pickle file
df = pd.read_pickle("../work/input/LSWMD_25519.pkl")

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="failureType", y="lotName")

# Set plot title and labels
plt.title("Lot Name vs Failure Type")
plt.xlabel("Failure Type")
plt.ylabel("Lot Name")

# Show the plot
plt.show()

