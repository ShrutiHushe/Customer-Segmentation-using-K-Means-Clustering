# Import the Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Data
data = pd.read_csv("customers_march24.csv")
print(data)

# Check for Null Data
print(data.isnull().sum())

# Features
features = data[["Annual_Income", "Spending_Score"]]

# Feature Scaling
mms = MinMaxScaler()
sfeatures = mms.fit_transform(features)

# Model
model = KMeans(n_clusters = 5, random_state = 9)
res =  model.fit_predict(sfeatures)
data["clusters"] = res
print(data)

# Prediction
anl_inc = float(input("Enter Annual Income\n"))
spen_scr = float(input("Enter Spending Score\n"))

d = [[anl_inc, spen_scr]]
sd = mms.transform(d)
ans = model.predict(sd)
print(ans)

if ans == 0:
	print("High Income low Spend")
elif ans == 1:
	print("Mid Income Mid Spend")
elif ans == 2:
	print("High Income High Spend")
elif ans == 3:
	print("Low Income High Spend")
elif ans == 4:
	print("Low Income Low Spend")
else:
	print("\n")


# Plotting 
d0 = data[data.clusters == 0]
d1 = data[data.clusters == 1]
d2 = data[data.clusters == 2]
d3 = data[data.clusters == 3]
d4 = data[data.clusters == 4]
plt.scatter(d0["Annual_Income"], d0["Spending_Score"], label="0")  #  High Income low Spend
plt.scatter(d1["Annual_Income"], d1["Spending_Score"], label="1")  #  Mid Income Mid Spend
plt.scatter(d2["Annual_Income"], d2["Spending_Score"], label="2")  #  High Income High Spend
plt.scatter(d3["Annual_Income"], d3["Spending_Score"], label="3")  #  Low Income High Spend
plt.scatter(d4["Annual_Income"], d4["Spending_Score"], label="4")  #  Low Income Low Spend
plt.legend()
plt.show()