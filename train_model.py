import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor

# Example data for training the model (importance, days_left, description_length)
X_train = np.array([[5, 2, 20], [3, 10, 15], [1, 5, 25], [4, 1, 30]])
y_train = np.array([9, 6, 4, 8])  # Priority scores based on user feedback

# Train the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'task_priority_model.pkl')

print("Model trained and saved as 'task_priority_model.pkl'")
