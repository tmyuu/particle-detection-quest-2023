import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the pickle file
df = pd.read_pickle("../work/input/LSWMD_25519.pkl")

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="failureType", y="dieSize")

# Set plot title and labels
plt.title("Die Size vs Failure Type")
plt.xlabel("Failure Type")
plt.ylabel("Die Size")

# Show the plot
plt.show()
