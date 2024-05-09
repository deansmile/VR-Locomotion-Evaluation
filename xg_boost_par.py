import numpy as np
import h5py
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data_from_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        DE_group = f['DE']
        DE_loaded = [DE_group[str(i)][:] for i in range(len(DE_group))]
        labels_loaded = f['labels'][:]
    
    result_arrays = []
    for array in DE_loaded:
        slice_32 = array[32]
        avg_0_5 = (array[0] + array[5]) / 2
        slice_1 = array[1]
        avg_2_13 = (array[2] + array[13]) / 2
        slice_40 = array[40]
        new_array = np.stack([slice_32, avg_0_5, slice_1, avg_2_13, slice_40], axis=0)
        result_arrays.append(new_array)
    
    DE_loaded = result_arrays
    max_time_length = max(de.shape[1] for de in DE_loaded)
    DE_padded = np.array([np.pad(de, ((0, 0), (0, max_time_length - de.shape[1]), (0, 0)), 'constant', constant_values=0) for de in DE_loaded])
    DE_flattened = np.array([de.reshape(-1) for de in DE_padded])
    labels_adjusted = labels_loaded + 1  # Adjust labels from [-1, 0, 1] to [0, 1, 2]
    
    return DE_flattened, labels_adjusted

# Load training and testing data
X_train, y_train = load_data_from_h5('train_data.h5')
X_test, y_test = load_data_from_h5('test_data.h5')

# Define and train the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_test)
overall_accuracy = accuracy_score(y_test, y_pred)
print(f'Overall XGBoost Model Accuracy: {overall_accuracy}')

# Compute class-based accuracy
cm = confusion_matrix(y_test, y_pred)
class_accuracies = cm.diagonal() / cm.sum(axis=1)
for i, acc in enumerate(class_accuracies):
    print(f'Accuracy for class {i}: {acc:.2f}')
