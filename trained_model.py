import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
df = pd.read_csv("house_price_tree.csv")

# Features and target
X = df[["size_m2", "bedrooms", "location_score"]]
y = df["price_category"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save pickle file
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as house_price_model.pkl")
