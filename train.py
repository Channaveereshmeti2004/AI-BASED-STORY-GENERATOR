import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import X, y, total_words, max_sequence_len

# Build model
model = Sequential()

# Embedding layer
model.add(
    Embedding(
        input_dim=total_words,
        output_dim=100,
        input_length=max_sequence_len - 1
    )
)

# First LSTM layer
model.add(
    LSTM(
        64,
        return_sequences=True
    )
)

# Dropout for regularization
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(32))

# Dense hidden layer
model.add(Dense(100, activation='relu'))

# Output layer
model.add(Dense(total_words, activation='softmax'))

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Show model summary
model.summary()

# Early stopping
early_stop = EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X,
    y,
    epochs=20,
    batch_size=64,
    verbose=1,
    callbacks=[early_stop]
)

# Save model
model.save("models/story_generator_model.h5")

print("\nModel Trained Successfully")

# Plot training accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()