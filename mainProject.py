from kfold_validation import validate
import sys
from plot import plot_training_history
emotion = sys.argv[2]
category  = sys.argv[3]
k = int(sys.argv[4])
model_name  = sys.argv[1]
train_loss , val_loss , train_acc , val_acc = validate(model_name, emotion ,category, k , 23)
history = {
    'train_loss' : train_loss , 
    'val_loss' : val_loss , 
    'train_acc' : train_acc , 
    'val_acc' : val_acc
}
plot_training_history(history)



