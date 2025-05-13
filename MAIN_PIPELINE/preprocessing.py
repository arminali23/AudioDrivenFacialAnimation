import numpy as np
import pickle

# === Extract PCA-Normalized Outputs ===

# Convert tensors to NumPy arrays
predicted_pca = pred_batch[0].cpu().numpy()
groundtruth_pca = gt_batch[0].cpu().numpy()


# === Load Preprocessing Parameters from File ===

with open("processed_data.pkl", "rb") as f:
    params = pickle.load(f)

# Extract required normalization and PCA parameters
expr_mean           = params['expression_mean']
expr_std            = params['expression_std']
pca_basis           = params['pca_components']
pca_internal_mean   = params['pca_internal_mean']
pca_mean            = params['pca_mean']
pca_std             = params['pca_std']


# === Define Function to Reconstruct Mesh from Normalized PCA ===

def inverse_normalize_pca(normalized_pca):
    """
    Reconstruct full expression mesh from PCA-normalized representation.
    """

    # Step 1: Undo z-score normalization applied after PCA
    denorm_pca = (normalized_pca * pca_std) + pca_mean

    # Step 2: Reverse PCA projection (project back into high-dimensional space)
    pca_projected = np.dot(denorm_pca, pca_basis)
    full_flat = pca_projected + pca_internal_mean

    # Step 3: Undo original expression normalization (std/mean)
    original_flat = (full_flat * expr_std) + expr_mean

    # Step 4: Reshape into original 3D mesh format
    return original_flat.reshape(-1, 5023, 3)


# === Recover Mesh Sequences from PCA Representation ===

predicted_expr = inverse_normalize_pca(predicted_pca)
groundtruth_expr = inverse_normalize_pca(groundtruth_pca)

print(predicted_expr.shape, groundtruth_expr.shape)

# add template mesh back to offsets to get full mesh vertices
predicted_mesh = predicted_expr * 1.5 + templates['FaceTalk_170809_00138_TA']
groundtruth_mesh = groundtruth_expr + templates['FaceTalk_170809_00138_TA']