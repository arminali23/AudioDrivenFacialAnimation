# load deepspeech audio features and index mapping
with open('/home/s5722127/MasterClass/traineddata/processed_audio_deepspeech.pkl', 'rb') as af:
    deepspeech_data = pickle.load(af, encoding='latin1')

with open('/home/s5722127/MasterClass/traineddata/subj_seq_to_idx.pkl', 'rb') as sf:
    index_map = pickle.load(sf, encoding='latin1')

with open('/home/s5722127/MasterClass/traineddata/templates.pkl', 'rb') as f:
    templates = pickle.load(f, encoding='latin1')

# show loaded subject names
if isinstance(deepspeech_data, dict):
    print("subjects:", list(deepspeech_data.keys()))


vertex_sequences = np.load('/home/s5722127/MasterClass/traineddata/data_verts.npy')

# Initialize empty lists to store paired data
audio_samples = []
expression_targets = []
sequence_labels = []

# Iterate over each subject in the index map
for subject_id in index_map.keys():
    subject_audio = deepspeech_data.get(subject_id)
    if subject_audio is None:
        continue

    subject_sequences = index_map[subject_id]

    # Iterate over each sentence for the current subject
    for sentence_id in subject_sequences.keys():
        sentence_audio = subject_audio.get(sentence_id)
        if sentence_audio is None:
            continue

        # Safely get the frame mapping and generate ordered indices
        frame_map = subject_sequences[sentence_id]
        ordered_indices = []
        for frame_key in sorted(frame_map):
            ordered_indices.append(frame_map[frame_key])

        # Retrieve mesh expression frames and audio features
        try:
            expr_seq = vertex_sequences[ordered_indices]
            audio_seq = sentence_audio['audio']
        except Exception as e:
            print(f"Skipping {subject_id}_{sentence_id} due to error: {e}")
            continue

        # Align both sequences to the same length
        sequence_length = min(len(expr_seq), len(audio_seq))
        expr_seq = expr_seq[:sequence_length]
        audio_seq = audio_seq[:sequence_length]

        # Compute mesh offset from template
        offset_seq = expr_seq - templates[subject_id][None, :]

        # Store the aligned data
        audio_samples.append(audio_seq)
        expression_targets.append(offset_seq)
        sequence_labels.append(f"{subject_id}_{sentence_id}")

# Display summary information
print(f"Total aligned sequences: {len(audio_samples)}")
print(f"Sample shapes: {audio_samples[0].shape}, {expression_targets[0].shape}, ID: {sequence_labels[0]}")

# === Setup PCA with Streaming Fit ===

n_dim = 1000
ipca = IncrementalPCA(n_components=n_dim)

# Instead of accumulating rows manually, create a generator-style flow
def generate_pca_chunks(sequences, scaler, chunk_size):
    buffer = []
    current_size = 0

    for seq in sequences:
        norm_seq = scaler.transform(seq.reshape(-1, 5023 * 3))
        buffer.append(norm_seq)
        current_size += norm_seq.shape[0]

        if current_size >= chunk_size:
            yield np.concatenate(buffer, axis=0)[:chunk_size]
            buffer.clear()
            current_size = 0

    if buffer:
        yield np.concatenate(buffer, axis=0)

# Feed chunks into partial_fit
for chunk in generate_pca_chunks(expression_targets, expr_scaler, n_dim):
    ipca.partial_fit(chunk)


# === Determine Optimal Component Count (99% Variance Retention) ===

explained = np.cumsum(ipca.explained_variance_ratio_)
num_components = next(i + 1 for i, val in enumerate(explained) if val >= 0.99)


# === Reduce All Sequences Using Trained PCA ===

# Using list comprehension with inline transformation
pca_expressions = [
    ipca.transform(expr_scaler.transform(seq.reshape(-1, 5023 * 3)))[:, :num_components]
    for seq in expression_targets
]


# === Save Key PCA Parameters for Future Use ===

pca_basis = ipca.components_[:num_components]
pca_mean = ipca.mean_
pca_var_ratio = ipca.explained_variance_ratio_[:num_components]

# === Combine and Normalize PCA-Projected Expression Data ===

# Concatenate all PCA-projected expression sequences into one large array
expr_pca = np.vstack(pca_expressions)  # shape: (total_frames, num_components)

# Calculate global mean and std for second-stage normalization
expr_pca_mean = np.mean(expr_pca, axis=0)
expr_pca_std = np.std(expr_pca, axis=0) + 1e-5  # avoid division by zero

# Normalize each PCA sequence using global stats
normalized_pca_expr = []
for seq in pca_expressions:
    normalized_seq = (seq - expr_pca_mean) / expr_pca_std
    normalized_pca_expr.append(normalized_seq)


# === Pack Processed Data into Dictionary ===

out = dict(
    audio=norm_audio,
    expression=normalized_pca_expr,
    audio_mean=audio_mu,
    audio_std=audio_sigma,
    expression_mean=expr_mean,
    expression_std=expr_std,
    pca_components=pca_basis,
    pca_internal_mean=pca_mean,
    explained_ratio=pca_var_ratio,
    pca_mean=expr_pca_mean,
    pca_std=expr_pca_std,
    sequence_labels=sequence_labels
)


# === Save Data Dictionary to Disk ===

output_path = 'processed_data.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(out, f)

with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)