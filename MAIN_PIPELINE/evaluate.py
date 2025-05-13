# === Load Pretrained Model from Checkpoint ===

model = AudioToExpressionTransformer(max_seq_len=30)
checkpoint = torch.load('best_model.pt', weights_only=True)
model.load_state_dict(checkpoint)
model.eval()


# === Define Evaluation Function for a Single Batch ===

def evaluate_one_batch(model, loader, device='cuda'):
    model.to(device)
    model.eval()

    # Fetch the first available batch from the loader
    try:
        mel_input, expr_target = next(iter(loader))
    except StopIteration:
        raise ValueError("Test loader is empty.")

    mel_input = mel_input.to(device)
    expr_target = expr_target.to(device)

    with torch.no_grad():
        prediction = model(mel_input)

    return prediction.cpu(), expr_target.cpu()


# === Run Evaluation on a Test Batch ===

pred_batch, gt_batch = evaluate_one_batch(model, test_loader)