import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, GlobalAveragePooling1D, Dense,
    LSTM, Dropout, Conv1D, MaxPooling1D, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

VOCAB_SIZE    = 10000
MAX_LENGTH    = 200
EMBEDDING_DIM = 32
BATCH_SIZE    = 64
EPOCHS        = 30

# ==============================================================================
# 1. GPU CHECK
# ==============================================================================

print("\n[ 1 / 7 ]  GPU CHECK")
print("-" * 60)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  ✅ GPU available: {gpus[0].name}")
else:
    print("  ❌ No GPU found. Go to Runtime > Change runtime type > GPU.")


# ==============================================================================
# 2. DATA LOADING & EXPLORATION
# ==============================================================================

print("\n[ 2 / 7 ]  DATA LOADING & EXPLORATION")
print("-" * 60)

(X_train_raw, y_train), (X_test_raw, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

lengths = [len(seq) for seq in X_train_raw]

print(f"  Training samples   : {len(X_train_raw):,}")
print(f"  Test samples       : {len(X_test_raw):,}")
print(f"  Class balance      — Positive: {y_train.sum():,}  |  Negative: {(y_train==0).sum():,}")
print()
print(f"  Sequence length stats (before padding):")
print(f"    Mean             : {np.mean(lengths):.1f} tokens")
print(f"    Median           : {np.median(lengths):.1f} tokens")
print(f"    Max              : {np.max(lengths)} tokens")
print(f"    Min              : {np.min(lengths)} tokens")
print(f"    % reviews ≤ {MAX_LENGTH} tokens : {(np.array(lengths) <= MAX_LENGTH).mean()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(lengths, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(MAX_LENGTH, color='red', linestyle='--', label=f'MAX_LENGTH = {MAX_LENGTH}')
axes[0].set_title('Distribution of Review Lengths (before padding)')
axes[0].set_xlabel('Number of tokens')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

counts = [(y_train == 0).sum(), (y_train == 1).sum()]
axes[1].bar(['Negative (0)', 'Positive (1)'], counts, color=['tomato', 'mediumseagreen'], edgecolor='white')
axes[1].set_title('Class Balance (Training Set)')
axes[1].set_ylabel('Number of samples')
axes[1].set_ylim(0, max(counts) * 1.2)
for i, v in enumerate(counts):
    axes[1].text(i, v + 100, str(v), ha='center', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

X_train = pad_sequences(X_train_raw, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test  = pad_sequences(X_test_raw,  maxlen=MAX_LENGTH, padding='post', truncating='post')

# ==============================================================================
# 3. MODEL ARCHITECTURES
# ==============================================================================

print("\n[ 3 / 7 ]  MODEL ARCHITECTURES")
print("-" * 60)

def build_baseline_model(embedding_dim=EMBEDDING_DIM):
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=embedding_dim, input_length=MAX_LENGTH),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1,  activation='sigmoid')
    ], name="Baseline_GAP")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(embedding_dim=EMBEDDING_DIM):
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=embedding_dim, input_length=MAX_LENGTH),
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1,  activation='sigmoid')
    ], name="CNN_1D")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(embedding_dim=EMBEDDING_DIM):
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=embedding_dim, input_length=MAX_LENGTH),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="LSTM")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_bidirectional_lstm_model(embedding_dim=EMBEDDING_DIM):
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=embedding_dim, input_length=MAX_LENGTH),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="Bi-LSTM")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

models = {
    "Baseline_GAP" : build_baseline_model(),
    "CNN_1D"       : build_cnn_model(),
    "LSTM"         : build_lstm_model(),
    "Bi-LSTM"      : build_bidirectional_lstm_model()
}

print(f"\n  {'Model':<16} {'Embedding dim':>14} {'Trainable params':>18}")
print(f"  {'-'*50}")
for name, model in models.items():
    model.build(input_shape=(None, MAX_LENGTH))
    print(f"  {name:<16} {EMBEDDING_DIM:>14} {model.count_params():>18,}")


# ==============================================================================
# 4. TRAINING
# ==============================================================================

print("\n[ 4 / 7 ]  TRAINING")
print("-" * 60)

def train_model(model, X_train, y_train):
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    return history

histories = {}
for name, model in models.items():
    print(f"\n  ── Training: {name} ──")
    histories[name] = train_model(model, X_train, y_train)