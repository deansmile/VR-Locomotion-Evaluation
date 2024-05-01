import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
# Step 1: Load data from the HDF5 file
file_path = 'data_3.h5'
with h5py.File(file_path, 'r') as f:
    DE_loaded = [f['DE'][str(i)][:] for i in range(len(f['DE']))]
    labels_loaded = f['labels'][:]

print(DE_loaded[0].shape)

# Finding the maximum time length for padding
max_time_length = max(de.shape[1] for de in DE_loaded)
print(max_time_length)
# Pad sequences to have the same time length
DE_padded = np.array([np.pad(de, ((0, 0), (0, max_time_length - de.shape[1]), (0, 0)), 'constant', constant_values=0) for de in DE_loaded])

# Flatten the data
DE_flattened = np.array([de.reshape(-1) for de in DE_padded])
print(DE_flattened[0].shape)

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(DE_flattened, labels_loaded, test_size=0.2, random_state=42)

# Step 3: Define and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Model Accuracy: {accuracy}')