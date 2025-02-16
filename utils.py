import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        Implements early stopping to halt training when validation loss stops improving.
        
        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            min_delta (float): Minimum improvement to reset patience counter.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        """
        Checks if early stopping should trigger.

        Args:
            val_loss (float): Current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def plot_loss(loss_history, save_path="loss_plot.png"):
    """
    Plots training and validation loss over epochs.

    Args:
        loss_history (dict): Dictionary containing lists of train/val losses.
        save_path (str): Path to save the loss plot image.
    """
    epochs = range(1, len(loss_history["train_emb"]) + 1)

    plt.figure(figsize=(10, 5))

    # Plot embedding losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history["train_emb"], label="Train Emb Loss", marker="o")
    plt.plot(epochs, loss_history["val_emb"], label="Val Emb Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Embedding Loss")
    plt.legend()

    # Plot generator losses
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_history["train_gen"], label="Train Gen Loss", marker="o")
    plt.plot(epochs, loss_history["val_gen"], label="Val Gen Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Generator Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
