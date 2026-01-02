from model_use.main import choose_model
import sys
from plot import plot_subject_dependet
import numpy as np
emotion = sys.argv[2]
category  = sys.argv[3]
k = int(sys.argv[4])
model_name  = sys.argv[1]
accuraceis = choose_model(model_name ,emotion , category, None , None  , subject_dependecy = 'subject_dependent')
print(f'''\n    average test accuracy :  {np.sum(accuraceis['test'])/23}
    average train accuracy : {np.sum(accuraceis['train'])/23}
''') 
# Added variance over per-subject accuracies
_test_accs = np.array(accuraceis['test'], dtype=float)
_train_accs = np.array(accuraceis['train'], dtype=float)
print(f"    variance test accuracy  : {np.var(_test_accs, ddof=1):.6f}")
print(f"    variance train accuracy : {np.var(_train_accs, ddof=1):.6f}")
plot_subject_dependet(accuraceis)
