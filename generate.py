import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load trained model
model = load_model("models/story_generator_model.h5")

# Max sequence length
max_sequence_len = 20

# Temperature sampling function
def sample_with_temperature(predictions, temperature=1.0):

    predictions = np.asarray(predictions).astype("float64")

    predictions = np.log(predictions + 1e-8) / temperature

    exp_preds = np.exp(predictions)

    predictions = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, predictions, 1)

    return np.argmax(probas)

# Input seed text
seed_text = input("Enter starting words: ")

# Words to generate
next_words = 50

# Creativity level
temperature = 0.8

generated_words = []

for _ in range(next_words):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding='pre'
    )

    predicted_probs = model.predict(token_list, verbose=0)[0]

    predicted_index = sample_with_temperature(
        predicted_probs,
        temperature
    )

    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break

    # Avoid repeating same word continuously
    if len(generated_words) > 0:
        if output_word == generated_words[-1]:
            continue

    generated_words.append(output_word)

    seed_text += " " + output_word

# Final output
print("\nGenerated Story:\n")
print(seed_text)

# Save generated story
with open(
    "outputs/generated_stories.txt",
    "a",
    encoding="utf-8"
) as file:

    file.write(seed_text + "\n\n")

print("\nStory saved successfully.")