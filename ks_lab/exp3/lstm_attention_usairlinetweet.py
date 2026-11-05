# %% [markdown]
# **ASSIGNMENT - 3**

# %% [markdown]
# # Implement an LSTM-based sentiment Analysis system with an Attention layer for US Airline Tweets Dataset

# %%
%pip install -q tensorflow numpy pandas matplotlib scikit-learn kagglehub

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

tf.keras.utils.set_random_seed(42)

# %%
import kagglehub
from kagglehub import KaggleDatasetAdapter
file_path = "Tweets.csv"
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "crowdflower/twitter-airline-sentiment",
  file_path,
)
df = df[['text', 'airline_sentiment']].dropna()
df = df.drop_duplicates(subset='text')   # duplicate tweets could leak between train and test
print(df.head(5))
print(df.columns)
print(df.shape)

# %%
def clean_text(t):
    t = t.lower()
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"n't", " not", t)                  # keep negation: didn't -> did not
    t = re.sub(r"'(s|re|ve|ll|d|m)\b", " ", t)
    t = re.sub(r"([!?])", r" \1 ", t)              # keep ! and ? as separate tokens
    t = re.sub(r"[^a-z!?\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df['clean_text'] = df['text'].apply(clean_text)
print(df[['text', 'clean_text']].head())

# %%
labels = pd.get_dummies(df['airline_sentiment']).astype(int)
class_names = list(labels.columns)
y = labels.values
X_text = df['clean_text'].values

X_temp, X_test_text, y_temp, y_test = train_test_split(
    X_text, y, test_size=0.20, random_state=42, stratify=y.argmax(axis=1)
)
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp.argmax(axis=1)
)

print("Train:", len(X_train_text), "Val:", len(X_val_text), "Test:", len(X_test_text))

# %%
vocab_size = 10000
max_length = 40

# filters='' so the ! and ? tokens kept by clean_text are not stripped again
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", filters='')
tokenizer.fit_on_texts(X_train_text)

def encode(texts):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

X_train = encode(X_train_text)
X_val = encode(X_val_text)
X_test = encode(X_test_text)

print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

# %% [markdown]
# ## Model with Attention layer

# %%
class AttentionPooling(layers.Layer):
    """Additive (Bahdanau-style) attention that collapses the time axis and respects the padding mask."""
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.supports_masking = True
        self.score_dense = Dense(units, activation='tanh')
        self.score_out = Dense(1, use_bias=False)

    def attention_weights(self, inputs, mask=None):
        scores = self.score_out(self.score_dense(inputs))           # (batch, time, 1)
        if mask is not None:
            mask = tf.cast(mask, scores.dtype)[..., tf.newaxis]
            scores += (1.0 - mask) * -1e9                            # padding tokens get ~0 weight
        return tf.nn.softmax(scores, axis=1)                         # weights over time sum to 1

    def call(self, inputs, mask=None):
        weights = self.attention_weights(inputs, mask)
        return tf.reduce_sum(weights * inputs, axis=1)               # (batch, features)

    def compute_mask(self, inputs, mask=None):
        return None  # time axis is gone, so no mask downstream

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# %%
embedding_dim = 128

inputs = Input(shape=(max_length,))
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)  # mask the padding zeros
x = SpatialDropout1D(0.3)(x)
x = Bidirectional(LSTM(64, return_sequences=True), name='bilstm')(x)  # keep every timestep for attention
x = AttentionPooling(64, name='attention')(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
# EarlyStopping restores the weights from the epoch with the lowest val_loss,
# so the evaluated model is the best one, not the last (most overfitted) one.
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=10,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# %%
best_epoch = int(np.argmin(history.history['val_loss']))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(history.history['accuracy'], label='Training Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy')
ax[0].set_title('Training vs Validation Accuracy'); ax[0].legend()

ax[1].plot(history.history['loss'], label='Training Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Loss')
ax[1].set_title('Training vs Validation Loss'); ax[1].legend()

for a in ax:
    a.axvline(best_epoch, color='gray', linestyle='--', label=f'Best epoch ({best_epoch + 1}) - weights restored')
    a.legend()

plt.show()

# %%
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss     :", loss)
print("Test Accuracy :", accuracy)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=class_names))

# %%
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(3), class_names)
plt.yticks(range(3), class_names)
plt.colorbar()

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.show()

# %%
def predict_sentiment(text):
    padded = encode([clean_text(text)])
    probs = model.predict(padded, verbose=0)[0]
    return class_names[int(np.argmax(probs))], probs

samples = [
    "The flight was amazing and the staff were very helpful",
    "Worst airline ever, my luggage was lost and nobody helped",
    "What time does the flight to Denver depart?"
]

for s in samples:
    label, probs = predict_sentiment(s)
    print(f"{s}\n  -> {label}  {dict(zip(class_names, probs.round(3)))}\n")

# %% [markdown]
# ## Attention weights visualization

# %%
# Which words did the model attend to?
bilstm_model = Model(model.input, model.get_layer('bilstm').output)
attention = model.get_layer('attention')

def show_attention(text):
    cleaned = clean_text(text)
    padded = encode([cleaned])
    seq_out = bilstm_model.predict(padded, verbose=0)
    mask = padded != 0
    weights = attention.attention_weights(seq_out, mask).numpy()[0, :, 0]

    n = int(mask.sum())
    words = [tokenizer.index_word.get(i, '<OOV>') for i in padded[0][:n]]
    label, probs = predict_sentiment(text)

    plt.figure(figsize=(max(6, n * 0.6), 2))
    plt.imshow(weights[:n][np.newaxis, :], cmap='Reds', aspect='auto')
    plt.xticks(range(n), words, rotation=45, ha='right')
    plt.yticks([])
    plt.title(f"Predicted: {label}")
    plt.colorbar()
    plt.show()

for s in samples:
    show_attention(s)


