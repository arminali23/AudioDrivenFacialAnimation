# === Setup Device and Model ===

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AudioToExpressionTransformer(max_seq_len=30).to(device)


# === Optimizer and Learning Rate Scheduler ===

optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

num_epochs = 100
total_steps = num_epochs * len(train_loader)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


# === Training Loop with Early Stopping ===

train_losses = []
val_losses = []
best_val = float('inf')
wait = 0
patience = 30

progress_bar = trange(num_epochs, desc="training", leave=True)

for epoch in progress_bar:
    # ---- Training Phase ----
    model.train()
    epoch_train_loss = 0.0

    for mel_batch, expr_batch in train_loader:
        mel_batch = mel_batch.to(device)
        expr_batch = expr_batch.to(device)

        optimizer.zero_grad()
        output = model(mel_batch)

        loss = torch.mean((output - expr_batch) ** 2)
        loss.backward()

        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)


    # ---- Validation Phase ----
    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for mel_batch, expr_batch in val_loader:
            mel_batch = mel_batch.to(device)
            expr_batch = expr_batch.to(device)

            predictions = model(mel_batch)
            loss = torch.mean((predictions - expr_batch) ** 2)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)


    # ---- Progress and Early Stopping ----
    progress_bar.set_description(
        f"[{epoch + 1}/{num_epochs}] train: {avg_train_loss:.4f} | val: {avg_val_loss:.4f}"
    )

    if avg_val_loss < best_val:
        best_val = avg_val_loss
        wait = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        wait += 1
        if wait >= patience:
            progress_bar.write(
                f"Early stopping triggered at epoch {epoch+1}: no improvement for {patience} epochs"
            )
            break


# === Visualize Training History ===

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train loss", linewidth=2)
plt.plot(val_losses, label="validation loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training and Validation Loss Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

