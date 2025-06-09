import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Set the dataset path

# Load the CSV file containing Solidity code and labels
df = pd.read_csv(r'C:\Users\Chatura Karankal\OneDrive\Desktop\smart_contract_ai\backend\dataset\SC_4label_5000.csv')  # Replace with actual filename
print("Total samples in dataset:", len(df))

# Inspect the columns
# Ensure the dataset contains 'contract_code' and 'label'
print(df.head())
print("Actual columns in the dataset:", df.columns.tolist())

# Preprocess the data
solidity_code_data = df['code'].tolist()  # List of Solidity contract code snippets
labels = df['label'].tolist()  # List of labels for each contract code

# Tokenizer setup
tokenizer = Tokenizer()
tokenizer.fit_on_texts(solidity_code_data)
sequences = tokenizer.texts_to_sequences(solidity_code_data)
maxlen = 200  # You can increase this based on the length of the contracts in the dataset
X = pad_sequences(sequences, maxlen=maxlen)

# Convert labels to numpy array
y = np.array(labels)

# Train/test split (80/20 for seen and unseen)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
# Build the LSTM model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=maxlen),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')  # Adjust the number of classes based on your dataset (4 classes in the example)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# Save the model and tokenizer for future use
model.save("models/lstm_model.h5")
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Print the training and testing accuracy
train_loss, train_acc = model.evaluate(X_train, y_train)
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Training Accuracy: {train_acc:.2f}")
print(f"Testing Accuracy: {test_acc:.2f}")