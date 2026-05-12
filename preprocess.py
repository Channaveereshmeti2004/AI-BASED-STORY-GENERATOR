import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
with open("dataset/stories.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()
# Reduce dataset size
    text = text[:3000000]

# Tokenizer
tokenizer = Tokenizer(
    num_words=8000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts([text])

# Save tokenizer
with open("models/tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Vocabulary size
total_words = min(8000, len(tokenizer.word_index) + 1)

# Input sequences
input_sequences = []

for line in text.split('\n'):

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Reduce memory usage
max_sequence_len = min(
    20,
    max(len(seq) for seq in input_sequences)
)

# Padding
input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding='pre'
    )
)

# Split data
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("\nPreprocessing Complete")
print("Vocabulary Size:", total_words)
print("Total Sequences:", len(input_sequences))
print("Max Sequence Length:", max_sequence_len)
print("Input Shape:", X.shape)
print("Output Shape:", y.shape)