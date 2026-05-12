#!/usr/bin/env python
"""
Quick script to create a minimal model file for testing
"""
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle

# Load tokenizer info
try:
    with open("models/tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    total_words = len(tokenizer.word_index) + 1
except:
    total_words = 5000  # fallback

max_sequence_len = 20

# Build a quick model
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len - 1))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(100, activation='relu'))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save the model
model.save("models/story_generator_model.h5")
print("✓ Model file created successfully at models/story_generator_model.h5")
