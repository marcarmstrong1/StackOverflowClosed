# Stack Overflow Question Classifier: Deep Learning with CNNs

## Project Context

This repository represents the next phase in our Natural Language Processing (NLP) curriculum.

In previous modules, we utilized traditional machine learning algorithms (Naive Bayes, Logistic Regression) and introduced semantic understanding via Word2Vec. This project marks the transition into **Deep Learning**.

Instead of manual feature engineering or standalone embeddings, this project utilizes **TensorFlow and Keras** to build an End-to-End Deep Learning pipeline. We treat the text classification problem using a **1D Convolutional Neural Network (CNN)**.

## Project Overview

The objective is to classify Stack Overflow questions into one of five categories based on their quality and content:

1.  **Open** (Valid questions)
2.  **Not a real question**
3.  **Off topic**
4.  **Not constructive**
5.  **Too localized**

### The Architecture

While CNNs are typically associated with Image Processing, they are highly effective for text classification.

[Image of 1D Convolutional Neural Network architecture for text classification]

The model implemented in this repository follows this architecture:

1.  **Input Layer:** Raw text data (Title + Body).
2.  **Embedding Layer:** Transforms integer-encoded words into dense vectors of size 64.
3.  **Convolutional Blocks (x3):** Three stacking layers of 1D Convolutions followed by Max Pooling. This allows the model to detect local patterns (n-grams) and reduce dimensionality.
4.  **Dense Layer:** A final fully connected layer that outputs the probability for the 5 classes.

## Prerequisites

To run this project, you will need the following libraries:

```text
tensorflow
pandas
nltk
seaborn
scikit-learn
matplotlib
```

## Data Structure

The script expects a file named `train-sample.csv` in the root directory. It requires the following columns:

  * `Title`: The title of the Stack Overflow post.
  * `BodyMarkdown`: The body text of the post.
  * `OpenStatus`: The target label (string format, e.g., "open", "off topic").

## Code Structure

The project consists of two main components:

  * **`main.py`** (The script): Handles data loading, orchestration of the training loop, and evaluation.
  * **`stackO_functions.py`**: A helper module containing specific logic for:
      * `remove_stopwords`: NLTK-based cleaning.
      * `train_val_split`: Splitting data for validation.
      * `fit_tokenizer` & `seq_and_pad`: Text vectorization.
      * `create_model`: Defines the Keras Sequential model.

## Performance

  * **Target Benchmark:** 60-70% Accuracy.
  * **Current Baseline:** \~64% Validation Accuracy.

## The Challenge

Your task is to improve upon this baseline CNN implementation. Consider the following modifications:

1.  **Architecture Change:** CNNs are great for local patterns, but Recurrent Neural Networks (RNNs) are often better for sequential context. Try implementing an **LSTM** (Long Short-Term Memory) or **GRU** (Gated Recurrent Unit) layer in place of the Convolutional layers.
2.  **Activation Function:** The current model uses a `sigmoid` activation on the final layer. For a multi-class problem where classes are mutually exclusive, `softmax` is arguably the standard. Test if this changes convergence speed or accuracy.
3.  **Pre-trained Embeddings:** The current model learns embeddings from scratch. Try injecting **GloVe** or **Word2Vec** weights into the embedding layer to see if transfer learning improves performance.
