# main.py

from model import AudioToExpressionTransformer
from dataset import LipSyncWindowDataset
from evaluate import evaluate
from train import train
from preprocessing import StandardScaler, IncrementalPCA
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from preprocessing import inverse_normalize_pca

import numpy as np
import torch
import pickle
import os

# === Parameters ===
WIN_SIZE = 30
STEP_SIZE = 2
BATCH_SIZE = 32
NUM_EPOCHS = 200
PATIENCE = 30

# === Load training data files ===
data_dir = '/home/s5722127/MasterClass/traineddata'
with open(os.path.join(data_dir, 'processed_audio_deepspeech.pkl'), 'rb') as af:
    deepspeech_data = pickle.load(af, encoding='latin1')
with open(os.path.join(data_dir, 'subj_seq_to_idx.pkl'), 'rb') as sf:
    index_map = pickle.load(sf, encoding='latin1')
with open(os.path.join(data_dir, 'templates.pkl'), 'rb') as f:
    templates = pickle.load(f, encoding='latin1')
vertex_sequences = np.load(os.path.join(data_dir, 'data_verts.npy'))

# === Preprocess and pair sequences ===
audio_samples = []
expression_targets = []
sequence_labels = []

for subject_id in index_map.keys():
    subject_audio = deepspeech_data.get(subject_id)
    if subject_audio is None:
        continue

    for sentence_id, frame_map in index_map[subject_id].items():
        sentence_audio = subject_audio.get(sentence_id)
        if sentence_audio is None:
            continue

        ordered_indices = [frame_map[k] for k in sorted(frame_map)]

        try:
            expr_seq = vertex_sequences[ordered_indices]
            audio_seq = sentence_audio['audio']
        except Exception as e:
            continue

        length = min(len(expr_seq), len(audio_seq))
        expr_seq = expr_seq[:length]
        audio_seq = audio_seq[:length]
        offset_seq = expr_seq - templates[subject_id][None, :]

        audio_samples.append(audio_seq)
        expression_targets.append(offset_seq)
        sequence_labels.append(f"{subject_id}_{sentence_id}")

# === Normalize audio ===
flat_audio = np.concatenate(audio_samples, axis=0).reshape(-1, 464)
audio_mu = flat_audio.mean(axis=0)
audio_sigma = flat_audio.std(axis=0) + 1e-5
norm_audio = [(seq.reshape(seq.shape[0], -1) - audio_mu) / audio_sigma for seq in audio_samples]

# === Normalize expression with StandardScaler ===
expr_scaler = StandardScaler()
for seq in expression_targets:
    expr_scaler.partial_fit(seq.reshape(-1, 5023 * 3))

# === Fit Incremental PCA ===
n_dim = 1000
ipca = IncrementalPCA(n_components=n_dim)

def generate_pca_chunks(sequences, scaler, chunk_size):
    buffer, current_size = [], 0
    for seq in sequences:
        norm_seq = scaler.transform(seq.reshape(-1, 5023 * 3))
        buffer.append(norm_seq)
        current_size += norm_seq.shape[0]
        if current_size >= chunk_size:
            yield np.concatenate(buffer, axis=0)[:chunk_size]
            buffer.clear(); current_size = 0
    if buffer:
        yield np.concatenate(buffer, axis=0)

for chunk in generate_pca_chunks(expression_targets, expr_scaler, n_dim):
    ipca.partial_fit(chunk)

# === Select number of PCA components (99% variance) ===
explained = np.cumsum(ipca.explained_variance_ratio_)
num_components = next(i + 1 for i, val in enumerate(explained) if val >= 0.99)

# === Transform expression sequences ===
pca_expressions = [
    ipca.transform(expr_scaler.transform(seq.reshape(-1, 5023 * 3)))[:, :num_components]
    for seq in expression_targets
]

# === Normalize PCA output ===
expr_pca = np.vstack(pca_expressions)
expr_pca_mean = np.mean(expr_pca, axis=0)
expr_pca_std = np.std(expr_pca, axis=0) + 1e-5
normalized_pca_expr = [(seq - expr_pca_mean) / expr_pca_std for seq in pca_expressions]

# === Save processed data ===
out = dict(
    audio=norm_audio,
    expression=normalized_pca_expr,
    audio_mean=audio_mu,
    audio_std=audio_sigma,
    expression_mean=expr_scaler.mean_,
    expression_std=np.sqrt(expr_scaler.var_) + 1e-5,
    pca_components=ipca.components_[:num_components],
    pca_internal_mean=ipca.mean_,
    explained_ratio=ipca.explained_variance_ratio_[:num_components],
    pca_mean=expr_pca_mean,
    pca_std=expr_pca_std,
    sequence_labels=sequence_labels
)

with open('processed_data.pkl', 'wb') as f:
    pickle.dump(out, f)

# === Load for training ===
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

train_audio, temp_audio, train_expr, temp_expr = train_test_split(data['audio'], data['expression'], test_size=0.2, random_state=42)
val_audio, test_audio, val_expr, test_expr = train_test_split(temp_audio, temp_expr, test_size=0.5, random_state=42)

train_dataset = LipSyncWindowDataset(train_audio, train_expr, win_size=WIN_SIZE, step=STEP_SIZE)
val_dataset   = LipSyncWindowDataset(val_audio, val_expr, win_size=WIN_SIZE, step=STEP_SIZE)
test_dataset  = LipSyncWindowDataset(test_audio, test_expr, win_size=WIN_SIZE, step=STEP_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Train the model ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AudioToExpressionTransformer(
    input_dim=464,
    output_dim=len(data['pca_mean']),
    max_seq_len=WIN_SIZE
).to(device)

train(model, train_loader, val_loader, NUM_EPOCHS, PATIENCE, device)

# === Evaluate one batch ===
pred_batch, gt_batch = evaluate_one_batch(model, test_loader, device)
print("Prediction shape:", pred_batch.shape)

predicted_expr = inverse_normalize_pca(pred_batch[0].cpu().numpy(), data)
groundtruth_expr = inverse_normalize_pca(gt_batch[0].cpu().numpy(), data)

subject_id = data['sequence_labels'][0].split('_')[0]
template_mesh = templates[subject_id]

predicted_mesh = predicted_expr * 1.5 + template_mesh
groundtruth_mesh = groundtruth_expr + template_mesh

anim = animate(predicted_mesh, groundtruth_mesh)
HTML(anim.to_jshtml())
