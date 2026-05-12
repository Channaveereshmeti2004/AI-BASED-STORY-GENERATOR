import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="AI Story Generator",
    page_icon="📖📝",
    layout="centered"
)

# -----------------------------------
# Load Tokenizer
# -----------------------------------

with open("models/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# -----------------------------------
# Load Model
# -----------------------------------

model = load_model("models/story_generator_model.h5")

# -----------------------------------
# Settings
# -----------------------------------

max_sequence_len = 20

# -----------------------------------
# Temperature Sampling
# -----------------------------------

def sample_with_temperature(predictions, temperature=1.0):

    predictions = np.asarray(predictions).astype("float64")

    predictions = np.log(predictions + 1e-8) / temperature

    exp_preds = np.exp(predictions)

    predictions = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, predictions, 1)

    return np.argmax(probas)

# -----------------------------------
# Story Generation Function
# -----------------------------------

def generate_story(seed_text, next_words, temperature):

    generated_words = []

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        predicted_probs = model.predict(
            token_list,
            verbose=0
        )[0]

        predicted_index = sample_with_temperature(
            predicted_probs,
            temperature
        )

        output_word = ""

        for word, index in tokenizer.word_index.items():

            if index == predicted_index:
                output_word = word
                break

        # Avoid repeating words
        if len(generated_words) > 0:

            if output_word == generated_words[-1]:
                continue

        generated_words.append(output_word)

        seed_text += " " + output_word

    return seed_text

# -----------------------------------
# UI
# -----------------------------------

st.title("📖 Generative AI Story Generator")

st.markdown(
    """
    Generate creative stories using an LSTM Neural Network trained from scratch.
    """
)

# Input text
seed_text = st.text_input(
    "Enter Starting Words",
    value="once upon"
)

# Story length
next_words = st.slider(
    "Story Length",
    min_value=20,
    max_value=200,
    value=50
)

# Creativity
temperature = st.slider(
    "Creativity Level",
    min_value=0.2,
    max_value=1.5,
    value=0.8,
    step=0.1
)

# Generate button
if st.button("Generate Story"):

    with st.spinner("Generating Story..."):

        story = generate_story(
            seed_text,
            next_words,
            temperature
        )

    st.subheader("Generated Story")

    st.write(story)

    # Download button
    st.download_button(
        label="Download Story",
        data=story,
        file_name="generated_story.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown(
    "Built using TensorFlow, LSTM, and Streamlit"
)