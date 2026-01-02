import matplotlib.pyplot as plt
import torch
from pathlib import Path


def plot_training_history(history, save_dir=None):
    """
    رسم نمودارهای loss و accuracy
    
    Parameters:
    -----------
    history : dict
        تاریخچه آموزش شامل train_loss, val_loss, train_acc, val_acc
    save_dir : Path or str, optional
        مسیر ذخیره نمودارها (اگر None باشد در مسیر فعلی ذخیره می‌شود)
    """
    # بررسی و تبدیل تنسور به numpy
    for key in history:
        if isinstance(history[key], torch.Tensor):
            history[key] = history[key].detach().cpu().numpy()
        elif isinstance(history[key], list):
            # تبدیل لیست به numpy array
            import numpy as np
            history[key] = np.array(history[key])
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        loss_path = save_dir / "loss_plot.png"
        acc_path = save_dir / "accuracy_plot.png"
    else:
        loss_path = "loss_plot.png"
        acc_path = "accuracy_plot.png"

    # --- نمودار Loss ---
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # --- نمودار Accuracy ---
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()


def plot_subject_dependet(accuracies, save_dir=None):
    """
    رسم نمودار accuracy برای هر subject
    
    Parameters:
    -----------
    accuracies : dict
        دیکشنری شامل 'train' و 'test' با لیست accuracy برای هر subject
    save_dir : Path or str, optional
        مسیر ذخیره نمودار (اگر None باشد در مسیر فعلی ذخیره می‌شود)
    """
    # بررسی و تبدیل تنسور به numpy
    import numpy as np
    for key in accuracies:
        if isinstance(accuracies[key], torch.Tensor):
            accuracies[key] = accuracies[key].detach().cpu().numpy()
        elif isinstance(accuracies[key], list):
            accuracies[key] = np.array(accuracies[key])
    
    subjects = range(1, len(accuracies['train']) + 1)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_path = save_dir / "subject_accuracy_plot.png"
    else:
        plot_path = "loss_plot.png"

    # --- نمودار Accuracy برای هر Subject ---
    plt.figure(figsize=(10, 6))
    plt.plot(subjects, accuracies['train'], label='Train Accuracy', marker='o')
    plt.plot(subjects, accuracies['test'], label='Test Accuracy', marker='s')
    plt.xlabel('Subject')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Subject')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()



