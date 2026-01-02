from dataset.main import data
import torch 
import os # os را برای چک کردن cuda اضافه کنید
from models_structures.simpleNN import model
from train import Trainer
import torch
import plot 
import matplotlib.pyplot as plt

# ____________DATA SET __________#
#the first step is to make datset ready for work 
test_person = [0 , 6 , 13, 19 , 21 ]
overlap = 0.4
time_len = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotion= 'vale'
label_method = 'binary'
batch_size = 160
data_type = torch.float32
my_dataset = data(test_person, overlap, time_len, device, emotion, label_method, batch_size, data_type)
train_loader = my_dataset.train_data()
test_loader = my_dataset.test_data()

#____Model______#
Model = model([1792, 64, 1])  # معماری دلخواه

#____trainer_______#
trainer = Trainer(
    model=Model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    optimizer_cls=torch.optim.Adam,
    lr=5e-4,
    epochs=50,
    checkpoint_path="eeg_checkpoint.pth",
    log_path="eeg_log.json"
)
#____fit_model_____#
history = trainer.fit()
#____plot_result___#
plot.plot_training_history(history)
  










