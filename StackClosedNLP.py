# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:30:03 2023

@author: Marcus
"""
#Imports
import pandas as pd
from stackO_functions import remove_stopwords, train_val_split, fit_tokenizer, seq_and_pad, create_model, print_cm

#Poor Practice but hardcoded variables
NUM_WORDS = 5000
EMBEDDING_DIM = 64
MAXLEN = 200
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SPLIT = .8

#Read in data and split into training and validation data
df = pd.read_csv("train-sample.csv")
df_train = df[['Title', 'BodyMarkdown', 'OpenStatus']]
df_train['OpenStatus']= df_train['OpenStatus'].map({'open': 0, 
                                                        'not a real question': 1, 
                                                        'off topic': 2, 
                                                        'not constructive': 3, 
                                                        'too localized': 4}) 
sentences = df_train["Title"] + ' ' + df_train["BodyMarkdown"]
target = df_train["OpenStatus"]
sentences = sentences.apply(remove_stopwords)
train_sentences, val_sentences, train_labels, val_labels = train_val_split(sentences, target, TRAINING_SPLIT)

#Fit the tokenizer
tokenizer = fit_tokenizer(train_sentences, NUM_WORDS, OOV_TOKEN)
word_index = tokenizer.word_index
print(f"Vocabulary contains {len(word_index)} words\n")
print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")

#Tokenize and pad the data
train_padded_seq = seq_and_pad(train_sentences, tokenizer, PADDING, MAXLEN)
val_padded_seq = seq_and_pad(val_sentences, tokenizer, PADDING, MAXLEN)
print(f"Padded training sequences have shape: {train_padded_seq.shape}\n")
print(f"Padded validation sequences have shape: {val_padded_seq.shape}")

#Create the model and fit
model = create_model(NUM_WORDS, EMBEDDING_DIM, MAXLEN)
history = model.fit(train_padded_seq, train_labels, epochs=2, validation_data=(val_padded_seq, val_labels))
  
t = model.predict(val_padded_seq)
t  = pd.DataFrame(t)
t["max"] = t.iloc[:,0:6].idxmax(axis = 1)

print_cm(val_labels, t)
