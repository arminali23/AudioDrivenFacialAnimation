import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# === Animate Predicted vs. Ground Truth 3D Mesh Side-by-Side ===

def animate(pred_seq, gt_seq, num_frames=None, interval=100, zoom=1.0,
                       elev=-90, azim=90, size=0.3):
    """
    Visual comparison between predicted and ground truth 3D face sequences.

    Parameters:
        pred_seq (array): Predicted vertex sequence (T, 5023, 3)
        gt_seq   (array): Ground truth vertex sequence (T, 5023, 3)
        num_frames (int): Number of frames to animate
        interval   (int): Delay between frames in ms
        zoom     (float): Scaling factor for display
        elev, azim (int): Camera view angles
        size     (float): Scatter point size
    """

    # Convert input to consistent array format
    pred = np.stack(pred_seq) if isinstance(pred_seq, (list, tuple)) else pred_seq
    gt   = np.stack(gt_seq)   if isinstance(gt_seq, (list, tuple))   else gt_seq

    # Limit number of frames if needed
    total_frames = min(len(pred), len(gt))
    num_frames = min(num_frames or total_frames, total_frames)

    # Apply scaling
    pred *= zoom
    gt   *= zoom

    # Set up the figure and axes
    fig = plt.figure(figsize=(10, 5))
    ax_pred = fig.add_subplot(1, 2, 1, projection='3d')
    ax_gt   = fig.add_subplot(1, 2, 2, projection='3d')

    for ax, label in [(ax_pred, "Prediction"), (ax_gt, "Ground Truth")]:
        ax.set_title(label)
        ax.set_axis_off()
        ax.set_xlim([-0.1, 0.1])
        ax.set_ylim([-0.1, 0.1])
        ax.set_zlim([-0.1, 0.1])
        ax.view_init(elev=elev, azim=azim)

    # Initialize scatter plots
    scatter_pred = ax_pred.scatter([], [], [], s=size)
    scatter_gt   = ax_gt.scatter([], [], [], s=size)

    # Define frame update function
    def update(frame_idx):
        frame_pred = pred[frame_idx]
        frame_gt   = gt[frame_idx]

        scatter_pred._offsets3d = (frame_pred[:, 0], frame_pred[:, 1], frame_pred[:, 2])
        scatter_gt._offsets3d   = (frame_gt[:, 0], frame_gt[:, 1], frame_gt[:, 2])
        return scatter_pred, scatter_gt

    # Build animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        blit=False
    )

    plt.close(fig)
    return anim

# animate prediction vs ground truth 
anim = animate(predicted_mesh, groundtruth_mesh, interval=100, zoom=1, size=1)
HTML(ani.to_jshtml())


def overlay_mesh_comparison(pred_seq, gt_seq, frame_idx=0, zoom=1.0, elev=-90, azim=90, size=0.5):
    
    # Get frame
    pred = pred_seq[frame_idx] * zoom
    gt   = gt_seq[frame_idx] * zoom

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Ground truth in blue
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=size, c='blue', label='Ground Truth')

    # Prediction in red
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=size, c='red', alpha=0.6, label='Prediction')

    ax.set_title(f"Frame {frame_idx} – Prediction vs Ground Truth")
    ax.set_axis_off()
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
# frame deformation over time
overlay_mesh_comparison(predicted_mesh, groundtruth_mesh, frame_idx=20)

def plot_temporal_mse(pred_seq, gt_seq):
    num_frames = min(len(pred_seq), len(gt_seq))
    mse_per_frame = [
        np.mean((pred_seq[i] - gt_seq[i]) ** 2)
        for i in range(num_frames)
    ]

    plt.figure(figsize=(8, 4))
    plt.plot(mse_per_frame, color='orange', linewidth=2)
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Squared Error")
    plt.title("Prediction vs Ground Truth Error Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_temporal_mse(predicted_mesh, groundtruth_mesh)
