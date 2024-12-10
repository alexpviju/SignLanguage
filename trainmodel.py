import numpy as np
import os
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

# Load sequences and labels
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            res = np.load(file_path, allow_pickle=True)
            
            # Debug: Check if res is empty or has unexpected shape
            if res.size == 0:
                print(f"Warning: Loaded empty array from {file_path}")
                continue  # Skip empty frames
            
            if len(res.shape) == 0:
                print(f"Warning: Loaded array with unexpected shape from {file_path}: {res.shape}")
                continue  # Skip if it's not an array with expected dimensions
            
            print(f"Loaded shape for {file_path}: {res.shape}")  # Debug line
            window.append(res)
        
        if window:  # Only append if window is not empty
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Warning: No valid frames loaded for action {action}, sequence {sequence}")

# Check shapes of loaded data
for i, window in enumerate(sequences):
    if window:  # Ensure window is not empty before checking shape
        print(f"Window {i} shape: {[frame.shape for frame in window]}")  # Debug line

# Ensure sequences are padded correctly
if sequences:  # Check if sequences is not empty
    max_sequence_length = max(len(window) for window in sequences)
    padded_sequences = []

    for window in sequences:
        if window:  # Ensure window is not empty
            # Check that all frames have the same number of features
            frame_shape = window[0].shape if len(window[0].shape) > 0 else None
            
            if frame_shape is None:
                print("Warning: Frame shape is None, skipping this window.")
                continue
            
            if len(window) < max_sequence_length:
                padding = np.zeros((max_sequence_length - len(window), frame_shape[0]))  # Pad with zeros
                padded_window = np.vstack((window, padding))
            else:
                padded_window = window[:max_sequence_length]  # Truncate if longer than max length
            
            padded_sequences.append(padded_window)

    # Convert to numpy array
    X = np.array(padded_sequences)
else:
    print("Warning: No sequences loaded, cannot create input array X.")

# Convert labels to categorical
y = to_categorical(labels).astype(int)

# Check final shapes
print(f"X shape: {X.shape}, y shape: {y.shape}")  # Debug line

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Model training setup
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(50,63)))  # Use X.shape for input size
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
