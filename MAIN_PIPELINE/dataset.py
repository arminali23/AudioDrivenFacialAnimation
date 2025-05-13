import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# === Custom Dataset for Sliding Window Lip Sync Sequences ===
class LipSyncWindowDataset(Dataset):
    def __init__(self, audio_sequences, expression_sequences, win_size=60, step=10):
        self.data_pairs = []

        # Generate paired audio-expression windows with sliding approach
        for audio_seq, expr_seq in zip(audio_sequences, expression_sequences):
            max_index = min(len(audio_seq), len(expr_seq)) - win_size + 1
            if max_index < 1:
                continue  # skip sequences too short for one window

            window_starts = range(0, max_index, step)

            for start in window_starts:
                end = start + win_size
                mel_segment = torch.tensor(audio_seq[start:end], dtype=torch.float32)
                expr_segment = torch.tensor(expr_seq[start:end], dtype=torch.float32)
                self.data_pairs.append((mel_segment, expr_segment))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]


# === Split Dataset Into Train, Validation, and Test Sets ===

# First: 80% train, 20% temp (for val + test)
train_audio, temp_audio, train_expr, temp_expr = train_test_split(
    data['audio'], data['expression'], test_size=0.2, random_state=42
)

# Then: split temp evenly into validation and test sets
val_audio, test_audio, val_expr, test_expr = train_test_split(
    temp_audio, temp_expr, test_size=0.5, random_state=42
)


# === Create Dataset Objects with Windowing Parameters ===

train_dataset = LipSyncWindowDataset(train_audio, train_expr, win_size=30, step=2)
val_dataset   = LipSyncWindowDataset(val_audio, val_expr, win_size=30, step=2)
test_dataset  = LipSyncWindowDataset(test_audio, test_expr, win_size=30, step=2)


# === Wrap in DataLoaders ===

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


# === Report Dataset Sizes ===

print(f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")