# Transformer Model for Machine Translation

This project is a complete implementation of the original "Attention Is All You Need" Transformer model, built from scratch using PyTorch. It's designed for neural machine translation (NMT) and includes all the core components, from data processing and tokenization to training and inference.

The model is trained on the **Opus Books** dataset to translate between English (`en`) and Dutch (`nl`), but it can be easily configured for other language pairs available in the dataset.

---

## 🚀 Features

* **Complete Transformer Architecture**: Full implementation of the encoder-decoder stack.
* **Multi-Head Self-Attention**: The core mechanism of the Transformer, implemented from scratch.
* **Positional Encoding**: Sinusoidal positional encodings to give the model a sense of sequence order.
* **Custom Learning Rate Scheduler**: A `NoamScheduler` that adjusts the learning rate as described in the original paper.
* **Hugging Face Integration**: Uses the `datasets` library to download data and the `tokenizers` library for building custom Word-Level tokenizers.
* **Dynamic Tokenization**: Automatically builds and saves tokenizers for the source and target languages if they don't exist.
* **Checkpointing**: Saves model and optimizer state after each epoch, allowing you to resume training.
* **Validation Loop**: Performs inference using greedy decoding on a validation set to monitor translation quality during training.
* **TensorBoard Logging**: Logs training loss for real-time monitoring.

---

## 📂 File Structure

The project is organized into several key files:

```bash
├── model.py            # Contains all the PyTorch modules for the Transformer architecture.
├── dataset.py          # Defines the BilingualDataset class for data loading and preprocessing.
├── config.py           # Central configuration file for all hyperparameters and paths.
├── train.py            # The main script to handle training, validation, and model saving.
├── weights/            # (Created automatically) Directory for saved model checkpoints.
├── tokenizer_en.json   # (Created automatically) Saved tokenizer for the source language.
├── tokenizer_nl.json   # (Created automatically) Saved tokenizer for the target language.
└── README.md           # This file.
```


* `model.py`: The heart of the project, defining classes like `MultiHeadAttentionBlock`, `EncoderBlock`, `DecoderBlock`, and the final `Transformer` model.
* `dataset.py`: Handles all data-related tasks. The `BilingualDataset` class takes raw text from the Hugging Face dataset and converts it into tokenized tensors with the necessary masks (`encoder_mask`, `decoder_mask`) for the model.
* `config.py`: A simple and clean way to manage all hyperparameters, such as batch size, learning rate, model dimensions, and language settings.
* `train.py`: Orchestrates the entire training process. It initializes the dataset, builds the tokenizers, sets up the model, optimizer, and loss function, and runs the main training and validation loops.

---

## 🛠️ Setup and Installation

Follow these steps to set up your environment and run the project.

**1. Clone the Repository**

```bash
git clone https://github.com/MonitSharma/Roadmap-to-LLMs.git
cd Roadmap-to-LLMs
```


**2. Create a Virtual Environment**

It's highly recommended to use a virtual environment to manage dependencies.

```bash
uv venv --python=python3.12
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

**3. Install Dependencies**

Download necessary pacakges from `requirements.txt` file

using 

```bash
uv pip install -r requirements.txt
```

## 🏃 How to Train the Model

**1. Configure Your Training Run**

Before starting, you can modify the config.py file to change the model's hyperparameters or the language pair.

```python
# config.py
def get_config():
    return {
        "batch_size" : 8,
        "num_epochs": 20,
        "lr" : 5e-5,
        "seq_len": 700,
        "d_model": 512,
        # ... other parameters
        "lang_src": "en",
        "lang_tgt": "nl",
        "preload": None, # or specify an epoch like "10" to resume
        # ...
    }
```



**2. Start Training**

Run the `train.py` script from your terminal 

```python
python train.py
```

The script will automatically:

1. Download the specified `opus_books` dataset.

2. Build and save word-level tokenizers for the source and target languages if they don't already exist.

3. Initialize the model, optimizer, and loss function.

4. Begin the training loop. A progress bar from `tqdm` will show the loss for each batch.

5. Run a validation loop at the end of each epoch, printing a few example translations.

6. Save a model checkpoint (`.pt` file) in the `weights/` directory.


**3. Monitor with TensorBoard**


You can visualize the training loss in real-time using TensorBoard.

```bash
tensorboard --logdir runs
```

Navigate to `http://localhost:6006/` in your browser to see the loss graph.

## 🧠 Model Architecture Explained

Think of the Transformer as a sophisticated information processing pipeline for translating sentences. It consists of two main parts: an **Encoder** and a **Decoder**. The Encoder's job is to read and understand the input sentence (e.g., in English), and the Decoder's job is to use that understanding to generate the translated sentence (e.g., in Dutch).



This entire pipeline is built from a few key, reusable components defined in `model.py`:

* **Input Embeddings & Positional Encoding**: First, we can't feed raw words to a neural network. The **`InputEmbeddings`** layer converts each word (token) into a numerical vector. Since the model processes all words at once and has no inherent sense of order, we add **`PositionalEncoding`**. This injects a special "timestamp" into each word's vector, telling the model its position in the sentence.

* **Multi-Head Attention (`MultiHeadAttentionBlock`)**: This is the Transformer's secret sauce. ✨ Instead of looking at a sentence word-by-word, this block allows every word to look at every *other* word in the sentence simultaneously. It calculates "attention scores" to determine which words are most important to understanding a given word. It's "Multi-Head" because it does this multiple times in parallel, with each "head" learning different types of relationships (e.g., one might learn grammatical links, another might learn semantic ones).

* **The Encoder (`Encoder` & `EncoderBlock`)**: The encoder is a stack of identical layers (`EncoderBlock`). Each layer runs the input sentence through the **Multi-Head Attention** block (so words can gather context from each other) and then through a simple **`FeedForward`** network for further processing.

* **The Decoder (`Decoder` & `DecoderBlock`)**: The decoder is also a stack of layers (`DecoderBlock`) and is a bit more complex. It has three main steps in each layer:
    1.  **Masked Multi-Head Attention**: It performs self-attention on the text it has generated so far. Critically, it's "masked" to prevent it from peeking at future words it hasn't predicted yet.
    2.  **Cross-Attention**: This is where the magic happens. The decoder takes what the encoder understood about the *source sentence* and combines it with the *target sentence* it's currently writing. It uses Multi-Head Attention to find the most relevant source words to focus on for predicting the next word.
    3.  **Feed-Forward Network**: Just like the encoder, it does some final processing on the combined information.

* **Final Output (`ProjectionLayer`)**: After passing through the entire decoder stack, the final vector is sent to a **`ProjectionLayer`**. This is a simple linear layer that maps the high-dimensional vector back to the size of our entire target vocabulary. A softmax function then turns these numbers into probabilities, and we pick the word with the highest probability as our translation.

* **Helper Modules (`LayerNormalization`, `ResidualConnection`)**: To keep the training process stable and efficient, the model uses **`LayerNormalization`** and **`ResidualConnection`** (also known as "Add & Norm") throughout the network. These act as traffic controllers, ensuring information flows smoothly without getting too large or too small.