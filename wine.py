# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sns

# %%
wine_df=pd.read_csv("winequality-red[1].csv", sep=";")
wine_df.head()
wine_df.tail()
pd.DataFrame(wine_df)

# %%
wine_df.info()

# %%
plt.scatter(wine_df['fixed acidity'], wine_df['quality'])
plt.xlabel("Fixed Acidity")
plt.ylabel("Wine Quality")
plt.title("Fixed Acidity vs Quality")
plt.show()

# %%
import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(wine_df.corr(), annot=True, cmap="coolwarm")
plt.show()

# %%
wine_df["quality"].unique

# %%
print(wine_df.columns)

# %%
wine_df["citric acid"]
wine_df["sulphates"]
wine_df["alcohol"]


# %%
wine_df.columns=wine_df.columns.str.strip().str.lower().str.replace(" ","_").str.replace("(","").str.replace(")","")
wine_df.columns

# %%
x=wine_df[["citric_acid","sulphates","alcohol"]]
y=wine_df["quality"]

# %%

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# %%
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(x_train, y_train)

# %%
y_pred=rf.predict(x_test)
y_pred

# %%
# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)

from sklearn.metrics import mean_squared_error, r2_score
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# %%
df_compare=pd.DataFrame({"Actual": y_test.values,"Predicted": y_pred})
df_compare.head(10)

# %%
import joblib
joblib.dump(rf, "wine_model.pkl")
model=joblib.load("wine_model.pkl")


# %%
new_wine = [[7.3,1.9, 0.57]]

prediction = model.predict(new_wine)

print("Predicted wine quality:", prediction)



# %%
import os
print(os.getcwd())


