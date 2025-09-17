# Next-word-predictor-LSTM-RNN
If u want to predict next word, go through this repo trained on Shakespeare-Hamlet  using LSTM -RNN.

This repository contains an **end-to-end Next Word Prediction model** trained on **Shakespeare's Hamlet** using an **LSTM Recurrent Neural Network (RNN)**.  
The model was trained locally and deployed as an interactive **Streamlit web app** for real-time text prediction.

---

##  Project Overview

- **Dataset**: *Hamlet by Shakespeare* (from NLTK corpus)
- **Model**: LSTM-based RNN
- **Parameters**: ~1.2 million trainable parameters
- **Accuracy**: ~81% on validation data
- **Language**: Python 3.10
- **Frameworks**: `tensorflow.keras`, `numpy`, `streamlit`
- **Deployment**: Streamlit app (`app.py`)

The model learns to predict the **next word** given a sequence of words, enabling text continuation and basic language modeling.

---

## 📂 Repository Structure
```
├── app.py # Streamlit app for deployment
├── experiments.ipynb # Data ingestion, cleaning, training, hyperparameter tuning
├── testtest.ipynb # Testing CUDA GPU + directory checks
├── hamlet.txt # Training dataset (Shakespeare’s Hamlet)
├── next_word_lstm.h5 # Trained LSTM model (~1.2M params)
├── tokenizer.pickle # Tokenizer (word → index mappings)
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

# Training the Model

All training and experimentation is done inside experiments.ipynb:

•Data ingestion from hamlet.txt
•Text cleaning & preprocessing
•Tokenization
•Sequence creation
•LSTM model training & hyperparameter tuning
•Model evaluation & saving (next_word_lstm.h5 + tokenizer.pickle)

The model was trained locally using TensorFlow (Keras API) with GPU support (CUDA verified in testtest.ipynb).

# Model Details

•Architecture: Multi-layer LSTM RNN
•Parameters: ~1.2 million
•Loss Function: Categorical Crossentropy
•Optimizer: Adam
•Accuracy: ~81% (achieved during training/validation)
•Input: Sequence of words (tokenized)
•Output: Most probable next word

📌 Notes

Model trained on a relatively small corpus (Hamlet), so predictions are Shakespearean in nature.
Streamlit app allows quick testing of the trained model.
You can extend the dataset for more robust predictions (e.g., full Shakespeare works or modern corpora).

