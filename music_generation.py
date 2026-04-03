import os
import random
import numpy as np
import matplotlib.pyplot as plt
from music21 import stream, note, tempo, instrument
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

print("=" * 60)
print("   CodeAlpha — Music Generation with AI")
print("=" * 60)

OUTPUT_FOLDER = "generated_music"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print("\nStep 1: Output folder ready")

MIDI_DATA = {
    "classical": [
        ["C4","E4","G4","C5","G4","E4","C4","D4","F4","A4","D5","A4"],
        ["D4","F4","A4","D5","A4","F4","D4","E4","G4","B4","E5","B4"],
        ["E4","G4","B4","E5","B4","G4","E4","F4","A4","C5","F5","C5"],
        ["F4","A4","C5","F5","C5","A4","F4","G4","B4","D5","G5","D5"],
        ["G4","B4","D5","G5","D5","B4","G4","A4","C5","E5","A5","E5"],
        ["C4","D4","E4","F4","G4","A4","B4","C5","B4","A4","G4","F4"],
        ["C5","B4","A4","G4","F4","E4","D4","C4","D4","E4","F4","G4"],
        ["E4","F4","G4","A4","B4","C5","D5","E5","D5","C5","B4","A4"],
        ["A4","B4","C5","D5","E5","F5","G5","A5","G5","F5","E5","D5"],
        ["G4","A4","B4","C5","D5","E5","F5","G5","F5","E5","D5","C5"],
    ],
    "jazz": [
        ["C4","E4","G4","B4","A4","G4","F4","E4","D4","C4","B3","C4"],
        ["D4","F4","A4","C5","B4","A4","G4","F4","E4","D4","C4","D4"],
        ["E4","G4","B4","D5","C5","B4","A4","G4","F4","E4","D4","E4"],
        ["F4","A4","C5","E5","D5","C5","B4","A4","G4","F4","E4","F4"],
        ["G4","B4","D5","F5","E5","D5","C5","B4","A4","G4","F4","G4"],
        ["A4","C5","E5","G5","F5","E5","D5","C5","B4","A4","G4","A4"],
        ["C4","E4","G4","A4","B4","A4","G4","E4","D4","C4","E4","G4"],
        ["D4","F4","A4","B4","C5","B4","A4","F4","E4","D4","F4","A4"],
        ["E4","G4","B4","C5","D5","C5","B4","G4","F4","E4","G4","B4"],
        ["F4","A4","C5","D5","E5","D5","C5","A4","G4","F4","A4","C5"],
    ],
    "pentatonic": [
        ["C4","D4","E4","G4","A4","G4","E4","D4","C4","E4","G4","A4"],
        ["G4","A4","C5","D5","E5","D5","C5","A4","G4","E4","D4","C4"],
        ["A4","C5","D5","E5","G5","E5","D5","C5","A4","G4","E4","D4"],
        ["D4","E4","G4","A4","C5","A4","G4","E4","D4","C4","A3","C4"],
        ["E4","G4","A4","C5","D5","C5","A4","G4","E4","D4","C4","D4"],
        ["C4","E4","G4","A4","C5","D5","C5","A4","G4","E4","C4","G3"],
        ["A3","C4","D4","E4","G4","A4","G4","E4","D4","C4","A3","G3"],
        ["G3","A3","C4","D4","E4","G4","A4","C5","A4","G4","E4","D4"],
    ],
    "minor": [
        ["A4","B4","C5","D5","E5","F5","G5","A5","G5","F5","E5","D5"],
        ["E4","F4","G4","A4","B4","C5","D5","E5","D5","C5","B4","A4"],
        ["D4","E4","F4","G4","A4","B4","C5","D5","C5","B4","A4","G4"],
        ["A3","B3","C4","D4","E4","F4","G4","A4","G4","F4","E4","D4"],
        ["C4","D4","E4","F4","G4","A4","B4","C5","A4","G4","F4","E4"],
        ["B3","C4","D4","E4","F4","G4","A4","B4","A4","G4","F4","E4"],
        ["G3","A3","B3","C4","D4","E4","F4","G4","F4","E4","D4","C4"],
    ],
}

all_notes = []
for genre, patterns in MIDI_DATA.items():
    for pattern in patterns:
        all_notes.extend(pattern)

print("Step 2: Music data loaded")
print("   Genres      :", list(MIDI_DATA.keys()))
print("   Total notes :", len(all_notes))


def preprocess(all_notes, sequence_length=8):
    print("\nStep 3: Preprocessing data...")

    unique_notes = sorted(list(set(all_notes)))
    n_vocab      = len(unique_notes)

    print("   Vocabulary size :", n_vocab)
    print("   Sequence length :", sequence_length)

    note_to_int = {n: i for i, n in enumerate(unique_notes)}
    int_to_note = {i: n for i, n in enumerate(unique_notes)}

    X_raw = []
    y_raw = []

    for i in range(len(all_notes) - sequence_length):
        seq_in  = all_notes[i : i + sequence_length]
        seq_out = all_notes[i + sequence_length]
        X_raw.append([note_to_int[n] for n in seq_in])
        y_raw.append(note_to_int[seq_out])

    X = np.reshape(X_raw, (len(X_raw), sequence_length, 1))
    X = X / float(n_vocab)
    y = np.array(y_raw)

    print("   Training samples:", X.shape[0])
    print("   X shape         :", X.shape)
    print("Step 3: Preprocessing complete!")

    return X, y, note_to_int, int_to_note, unique_notes, n_vocab


SEQUENCE_LENGTH = 8
X, y, note_to_int, int_to_note, unique_notes, n_vocab = preprocess(
    all_notes, SEQUENCE_LENGTH
)


def build_lstm_model(input_shape, n_vocab):
    print("\nStep 4: Building LSTM model...")

    model = Sequential([
        LSTM(512, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(256, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(128, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation="relu"),
        Dropout(0.2),

        Dense(128, activation="relu"),
        Dropout(0.2),

        Dense(n_vocab, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    print("Step 4: Model built! Parameters:", model.count_params())
    model.summary()
    return model


model = build_lstm_model(
    input_shape=(X.shape[1], X.shape[2]),
    n_vocab=n_vocab
)


def train_model(model, X, y):
    print("\nStep 5: Training the model...")

    checkpoint_path = os.path.join(OUTPUT_FOLDER, "best_model.keras")

    callbacks = [
        EarlyStopping(
            monitor="loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="loss",
            save_best_only=True,
            verbose=0
        )
    ]

    history = model.fit(
        X, y,
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("Step 5: Training complete!")
    print("   Final loss     :", round(history.history["loss"][-1], 4))
    print("   Final accuracy :", round(history.history["accuracy"][-1] * 100, 1), "%")
    return history


history = train_model(model, X, y)


def plot_training(history):
    print("\nStep 6: Saving training plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSTM Music Generation - Training Results", fontsize=14)

    axes[0].plot(history.history["loss"], color="#2c3e90", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], color="#27ae60", linewidth=2)
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_FOLDER, "training_results.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print("Step 6: Plot saved to", plot_path)


plot_training(history)


def generate_notes(model, all_notes, note_to_int, int_to_note,
                   unique_notes, n_vocab, num_notes=80, temperature=0.8):
    print("\nStep 7: Generating music...")

    start_idx   = random.randint(0, len(all_notes) - SEQUENCE_LENGTH - 1)
    seed        = all_notes[start_idx : start_idx + SEQUENCE_LENGTH]
    generated   = list(seed)
    current_seq = [note_to_int[n] for n in seed]

    print("   Seed sequence :", seed)

    for _ in tqdm(range(num_notes), desc="   Generating notes"):
        inp   = np.array(current_seq, dtype=float) / float(n_vocab)
        inp   = inp.reshape(1, SEQUENCE_LENGTH, 1)

        probs = model.predict(inp, verbose=0)[0]
        probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)

        next_idx  = np.random.choice(len(probs), p=probs)
        next_note = int_to_note[next_idx]

        generated.append(next_note)
        current_seq.append(next_idx)
        current_seq = current_seq[1:]

    print("Step 7: Generated", len(generated), "notes!")
    return generated


def notes_to_midi(generated_notes, filename, bpm=120, note_duration=0.5):
    print("\nSaving:", filename)

    music_stream = stream.Stream()
    music_stream.append(tempo.MetronomeMark(number=bpm))
    music_stream.append(instrument.Piano())

    for n in generated_notes:
        try:
            new_note = note.Note(n)
            new_note.duration.quarterLength = note_duration
            music_stream.append(new_note)
        except Exception:
            continue

    midi_path = os.path.join(OUTPUT_FOLDER, filename)
    music_stream.write("midi", fp=midi_path)
    print("Saved to", midi_path)
    return midi_path


print("\nGenerating 3 songs...")

notes1 = generate_notes(model, all_notes, note_to_int, int_to_note,
                        unique_notes, n_vocab, num_notes=80, temperature=0.8)
notes_to_midi(notes1, "song_classical.mid", bpm=90, note_duration=0.5)

notes2 = generate_notes(model, all_notes, note_to_int, int_to_note,
                        unique_notes, n_vocab, num_notes=80, temperature=1.0)
notes_to_midi(notes2, "song_jazz.mid", bpm=140, note_duration=0.25)

notes3 = generate_notes(model, all_notes, note_to_int, int_to_note,
                        unique_notes, n_vocab, num_notes=60, temperature=0.5)
notes_to_midi(notes3, "song_ambient.mid", bpm=60, note_duration=1.0)

txt_path = os.path.join(OUTPUT_FOLDER, "note_sequence.txt")
with open(txt_path, "w") as f:
    f.write("Generated Note Sequence\n")
    f.write("=" * 40 + "\n\n")
    f.write(" -> ".join(notes1))
    f.write("\n\nTotal notes: " + str(len(notes1)) + "\n")
print("Note sequence saved to", txt_path)

print("\n" + "=" * 60)
print("   MUSIC GENERATION COMPLETE!")
print("=" * 60)
print("Output folder  :", OUTPUT_FOLDER)
print("Files created  :")
print("   song_classical.mid")
print("   song_jazz.mid")
print("   song_ambient.mid")
print("   training_results.png")
print("   note_sequence.txt")
print("\nHow to play:")
print("   Go to generated_music/ folder")
print("   Double click any .mid file")
print("   Opens in VLC or Windows Media Player")
print("=" * 60)
