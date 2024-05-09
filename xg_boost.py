import numpy as np
import h5py
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Step 1: Load data from the HDF5 file
file_path = 'data_5.h5'
with h5py.File(file_path, 'r') as f:
    DE_loaded = [f['DE'][str(i)][:] for i in range(len(f['DE']))]
    labels_loaded = f['labels'][:]

result_arrays = []

for array in DE_loaded:
    # Selecting the slices based on the specified indices
    slice_32 = array[32]  # 32nd slice, index is 31
    avg_0_5 = (array[0] + array[5]) / 2  # Average of 0th and 5th slice
    slice_1 = array[1]  # 2nd slice, index is 1
    avg_2_13 = (array[2] + array[13]) / 2  # Average of 2nd and 13th slice
    slice_40 = array[40]  # 41st slice, index is 40

    # Combining the selected slices into a new array
    new_array = np.stack([slice_32, avg_0_5, slice_1, avg_2_13, slice_40], axis=0)
    result_arrays.append(new_array)

DE_loaded = result_arrays
# Finding the maximum time length for padding
max_time_length = max(de.shape[1] for de in DE_loaded)
print(max_time_length)
# Pad sequences to have the same time length
DE_padded = np.array([np.pad(de, ((0, 0), (0, max_time_length - de.shape[1]), (0, 0)), 'constant', constant_values=0) for de in DE_loaded])

# Flatten the data
DE_flattened = np.array([de.reshape(-1) for de in DE_padded])
print(DE_flattened[0].shape)

labels_adjusted = labels_loaded + 1
# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(DE_flattened, labels_adjusted, test_size=0.2, random_state=42)

# Step 3: Define and train the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f'XGBoost Model Accuracy: {accuracy}')

# Step 4: Evaluate the model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'XGBoost Model Accuracy: {accuracy}')
