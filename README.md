# 📖 AI-Based Story Generator using LSTM Neural Networks

A Generative AI project that generates meaningful and human-like stories using an LSTM (Long Short-Term Memory) neural network trained completely from scratch without using pretrained transformer models or external APIs.

---

# 🚀 Project Overview

This project demonstrates the practical implementation of Generative AI and Natural Language Processing (NLP) using Deep Learning techniques.

The system learns sequential language patterns from the TinyStories dataset and generates creative stories based on user-provided prompts.

A user-friendly Streamlit web interface allows users to:

* Enter custom prompts
* Generate stories in real time
* Control creativity level
* Control story length
* Download generated stories

---

# 🧠 Features

✅ Story generation using LSTM neural networks
✅ Model trained completely from scratch
✅ No pretrained transformer APIs used
✅ TinyStories dataset integration
✅ Temperature-based creativity control
✅ Streamlit web interface
✅ Download generated stories
✅ Repetition filtering and preprocessing
✅ TensorFlow and Keras implementation

---

# 🛠️ Technologies Used

| Technology     | Purpose                   |
| -------------- | ------------------------- |
| Python         | Programming Language      |
| TensorFlow     | Deep Learning Framework   |
| Keras          | Neural Network API        |
| Streamlit      | Web Application Interface |
| NumPy          | Numerical Computation     |
| Matplotlib     | Training Visualization    |
| NLP Techniques | Text Processing           |

---

# 🏗️ System Architecture

```text
Input Prompt
      ↓
Text Preprocessing
      ↓
Tokenization
      ↓
Embedding Layer
      ↓
LSTM Neural Network
      ↓
Next Word Prediction
      ↓
Story Generation
      ↓
Streamlit Web Interface
```

---

# 📂 Project Structure

```text
story_generator/
│
├── dataset/
│   └── stories.txt
│
├── models/
│   ├── tokenizer.pkl
│   └── story_generator_model.h5
│
├── outputs/
│   ├── accuracy_graph.png
│   ├── loss_graph.png
│   └── generated_stories.txt
│
├── preprocess.py
├── train.py
├── generate.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Bharghava-teja/AI-BASED-STORY-GENERATOR.git
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd AI-BASED-STORY-GENERATOR
```

---

## 3️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

---

## 4️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

## Preprocess Dataset

```bash
python preprocess.py
```

---

## Train Model

```bash
python train.py
```

---

## Generate Stories (Terminal)

```bash
python generate.py
```

---

## Launch Streamlit Web App

```bash
streamlit run app.py
```

---

# 🎯 Model Details

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Model Type    | LSTM                            |
| Dataset       | TinyStories                     |
| Optimizer     | Adam                            |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs        | 20                              |
| Batch Size    | 64                              |
| Framework     | TensorFlow/Keras                |

---

# 📊 Training Results

* Successfully trained LSTM neural network from scratch
* Achieved stable loss reduction during training
* Generated meaningful and readable stories
* Implemented creativity control using temperature sampling

---

# 🖥️ Streamlit Interface

The web interface provides:

* Prompt input box
* Story length slider
* Creativity level slider
* Generate story button
* Download generated story feature

---

# 📌 Sample Generated Output

### Input Prompt

```text
Once upon
```

### Generated Story

```text
Once upon a time there lived a brave little boy who wanted to explore the magical forest near his village.
```

---

# 🔮 Future Enhancements

* Transformer-based story generation
* Multi-language support
* Grammar correction
* Voice-based storytelling
* Cloud deployment
* Mobile application integration

---

# 📚 Learning Outcomes

* Generative AI concepts
* Natural Language Processing
* LSTM Neural Networks
* Deep Learning model training
* Streamlit deployment
* Sequence prediction techniques

---

# 👨‍💻 Team Members

* B Bharath Sai
* C H BHARGHAVATEJA VARDHAN
* PRAKASH K
* SHUBHAM YADAV

### Branch

CSE-AI, 8th Semester

---

# ⭐ Conclusion

This project demonstrates how Generative AI and LSTM-based neural networks can be used to generate meaningful stories using custom-trained models without relying on pretrained transformer APIs.

The system successfully integrates NLP, deep learning, and web deployment into a complete AI-based application.

---

# 📄 License

This project is developed for educational and academic purposes.
