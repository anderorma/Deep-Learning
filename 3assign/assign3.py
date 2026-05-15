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


# ==============================================================================
# 5. RESULTS
# ==============================================================================

print("\n[ 5 / 7 ]  RESULTS SUMMARY")
print("-" * 60)
print(f"\n  {'Model':<16} {'Test Accuracy':>14} {'Test Loss':>12}")
print(f"  {'-'*44}")

results = {}
for name, model in models.items():
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    results[name] = {'accuracy': acc, 'loss': loss}
    print(f"  {name:<16} {acc:>13.4f} {loss:>12.4f}")


# ==============================================================================
# 6. VISUALIZATIONS
# ==============================================================================

print("\n[ 6 / 7 ]  VISUALIZATIONS")
print("-" * 60)

# --- Validation curves ---
print("\n  Plotting validation curves...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for name, history in histories.items():
    axes[0].plot(history.history['val_accuracy'], label=name)
    axes[1].plot(history.history['val_loss'],     label=name)

axes[0].set_title('Validation Accuracy — All Models')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Validation Loss — All Models')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- ROC Curves ---
print("\n  Plotting ROC curves...")
plt.figure(figsize=(9, 7))
for name, model in models.items():
    y_pred = model.predict(X_test, verbose=0).ravel()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# --- Confusion Matrices ---
print("\n  Plotting confusion matrices...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, (name, model) in enumerate(models.items()):
    y_pred_classes = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f'Confusion Matrix: {name}')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')
    axes[i].set_xticklabels(['Negative', 'Positive'])
    axes[i].set_yticklabels(['Negative', 'Positive'])
plt.tight_layout()
plt.show()

# --- Accuracy Bar Chart ---
print("\n  Plotting accuracy bar chart...")
names  = list(results.keys())
accs   = [results[n]['accuracy'] for n in names]
colors = ['steelblue', 'mediumseagreen', 'tomato', 'mediumpurple']

plt.figure(figsize=(9, 5))
bars = plt.bar(names, accs, color=colors, edgecolor='white', width=0.5)
plt.ylim(min(accs) - 0.02, max(accs) + 0.02)
plt.title('Test Accuracy Comparison — All Models')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f'{acc:.4f}', ha='center', va='bottom', fontweight='bold'
    )
plt.tight_layout()
plt.show()


# ==============================================================================
# 7. IMPROVEMENT — LSTM & Bi-LSTM with EMBEDDING_DIM=64
# ==============================================================================

print("\n[ 7 / 7 ]  IMPROVEMENT EXPERIMENT — EMBEDDING DIM 32 vs 64")
print("-" * 60)
print("\n  Hypothesis: LSTM instability may be caused by insufficient")
print("  embedding dimensions. Re-training LSTM & Bi-LSTM with dim=64.\n")

EMBEDDING_DIM_V2 = 64

models_v2 = {
    "LSTM_64"    : build_lstm_model(embedding_dim=EMBEDDING_DIM_V2),
    "Bi-LSTM_64" : build_bidirectional_lstm_model(embedding_dim=EMBEDDING_DIM_V2)
}

print(f"\n  {'Model':<16} {'Embedding dim':>14} {'Trainable params':>18}")
print(f"  {'-'*50}")
for name, model in models_v2.items():
    model.build(input_shape=(None, MAX_LENGTH))
    print(f"  {name:<16} {EMBEDDING_DIM_V2:>14} {model.count_params():>18,}")

histories_v2 = {}
for name, model in models_v2.items():
    print(f"\n  ── Training: {name} (embedding_dim=64) ──")
    histories_v2[name] = train_model(model, X_train, y_train)

print(f"\n  {'Model':<16} {'Test Accuracy':>14} {'Test Loss':>12}")
print(f"  {'-'*44}")

results_v2 = {}
for name, model in models_v2.items():
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    results_v2[name] = {'accuracy': acc, 'loss': loss}
    print(f"  {name:<16} {acc:>13.4f} {loss:>12.4f}")

# --- Comparison: dim=32 vs dim=64 ---
print("\n  Plotting 32 vs 64 comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(histories["LSTM"].history['val_accuracy'],       label='LSTM (dim=32)',    linestyle='--', color='tomato')
axes[0].plot(histories_v2["LSTM_64"].history['val_accuracy'], label='LSTM (dim=64)',    linestyle='-',  color='tomato')
axes[0].plot(histories["Bi-LSTM"].history['val_accuracy'],       label='Bi-LSTM (dim=32)', linestyle='--', color='mediumpurple')
axes[0].plot(histories_v2["Bi-LSTM_64"].history['val_accuracy'], label='Bi-LSTM (dim=64)', linestyle='-',  color='mediumpurple')
axes[0].set_title('Validation Accuracy — Embedding dim 32 vs 64')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

all_names  = ['LSTM\n(dim=32)', 'LSTM\n(dim=64)', 'Bi-LSTM\n(dim=32)', 'Bi-LSTM\n(dim=64)']
all_accs   = [
    results["LSTM"]['accuracy'],
    results_v2["LSTM_64"]['accuracy'],
    results["Bi-LSTM"]['accuracy'],
    results_v2["Bi-LSTM_64"]['accuracy']
]
bar_colors = ['tomato', 'salmon', 'mediumpurple', 'plum']
bars = axes[1].bar(all_names, all_accs, color=bar_colors, edgecolor='white', width=0.5)
axes[1].set_ylim(min(all_accs) - 0.02, max(all_accs) + 0.02)
axes[1].set_title('Test Accuracy — Embedding dim 32 vs 64')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, all_accs):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f'{acc:.4f}', ha='center', va='bottom', fontweight='bold'
    )

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("   EXPERIMENT COMPLETE")
print("=" * 60)