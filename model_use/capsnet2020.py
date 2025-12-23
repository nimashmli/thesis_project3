from dataset.main import data
import torch 
import torch.nn as nn 
import os # os را برای چک کردن cuda اضافه کنید
from models_structures.capsnet2020 import model
from train import Trainer
import torch
from dataset.main import data , data_for_subject_dependet
from train import Trainer
import random
from functions import k_fold_data_segmentation
from  torch.utils.data import DataLoader , TensorDataset
import numpy as np 
from run_utils import ensure_dir, save_json, load_json


def loss_fn (v , y , landa=0.5 , m_plus=0.9 , m_mines=0.1) :  #v:  (B, M) y:(B)
    relu = nn.ReLU()
    total_loss  = 0 
    for i in range(v.shape[1]) : 
        T = (y == i ).float()
        Loss =  T * (relu(m_plus - v[: , i]))**2 + landa*(1-T)*(relu(v[: , i] -  m_mines))**2
        total_loss += Loss
    return total_loss.sum()


#____Model______#                          categy ; binary or 5category
def create_model(test_person , emotion,category , fold_idx , run_dir=None ) : 
    overlap = 0
    time_len = 1 
    num_filter =256
    num_channel = 14 
    caps_len = 8
    out_dim= 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if category == 'binary'  :
        output_dim = 2 
    elif category == '5category' :
        output_dim = 5
    num_emotions = output_dim
    batch_size =256
    data_type = torch.float32
    my_dataset = data(test_person, overlap, time_len, device, emotion, category, batch_size, data_type)
    train_loader = my_dataset.train_data()
    test_loader = my_dataset.test_data()
    Model = model (num_filter, 128* time_len, caps_len, num_emotions, out_dim)
    unique_Loss_fn = lambda v , y : loss_fn(v , y)
    #____trainer_______#
    # class weights for imbalance
    y_train = my_dataset.y_train
    class_count = torch.bincount(y_train.long())
    class_count = class_count + (class_count == 0).long()
    weights = (1.0 / class_count.float())
    weights = weights / weights.sum() * len(class_count)

    base_dir = run_dir or "."
    ensure_dir(base_dir)
    checkpoint_path = os.path.join(base_dir, "checkpoint.pth")
    log_path = os.path.join(base_dir, "log.json")
    status_path = os.path.join(base_dir, "status.json")

    trainer = Trainer(
        model=Model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        label_method=category,
        optimizer_cls=torch.optim.Adam,
        lr=2e-5,
        epochs=30,
        loss_fn = unique_Loss_fn, 
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        status_path=status_path,
    )
    #____fit_model_____#
    return  trainer.fit()


def subject_dependent_validation (emotion ,category, fold_idx , k=5, run_dir=None) : 
    num_filter =32
    num_channel = 14 
    caps_len = 4
    out_dim= 8
    overlap = 0
    time_len = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if category == 'binary'  :
        output_dim = 2 
    elif category == '5category' :
        output_dim = 5
    num_emotions = output_dim
    batch_size = 128
    data_type = torch.float32
    accuracies_on_subjects  = {
        'train' : [] , 
        'test' : []
    } 
    status_file = os.path.join(run_dir or ".", "status.json")
    history_file = os.path.join(run_dir or ".", "history.json")
    
    # لود کردن نتایج قبلی اگر وجود داشته باشد
    if os.path.exists(history_file):
        prev_history = load_json(history_file, {})
        if 'train' in prev_history and 'test' in prev_history:
            accuracies_on_subjects['train'] = list(prev_history['train'])
            accuracies_on_subjects['test'] = list(prev_history['test'])
    
    data = data_for_subject_dependet(overlap , time_len ,emotion ,category ,data_type , device  )
    total_subjects = len(data.data)
    status = load_json(status_file, default={
        "mode": "subject_dependent",
        "status": "running",
        "current_subject": 0,
        "current_fold": 0,
        "total_subjects": total_subjects,
        "total_folds": k,
    })
    
    # همیشه total_subjects و total_folds را از دیتاست به‌روزرسانی کن
    status["total_subjects"] = total_subjects
    status["total_folds"] = k
    
    start_subject = status.get("current_subject", 0)
    start_fold = status.get("current_fold", 0)

    for person_num, (x , y) in enumerate(data.data) :
        # اگر سوژه قبلاً کامل شده، نتایج را از history بخوان
        if person_num < start_subject:
            # اگر نتایج در history موجود نیست، از پوشه subject بخوان
            if person_num >= len(accuracies_on_subjects['train']):
                subj_dir = os.path.join(run_dir or ".", f"subject_{person_num}")
                subj_status_file = os.path.join(subj_dir, "status.json")
                if os.path.exists(subj_status_file):
                    subj_status = load_json(subj_status_file, {})
                    if 'train_acc' in subj_status and 'test_acc' in subj_status:
                        accuracies_on_subjects['train'].append(subj_status['train_acc'])
                        accuracies_on_subjects['test'].append(subj_status['test_acc'])
            continue
        
        # تعیین فولد شروع برای این سوژه
        if person_num == start_subject:
            fold_start = start_fold
        else:
            fold_start = 0
        len_data = x.shape[0]
        fold_number = len_data//k 
        all_x = [x[fold_number*i : min(fold_number*(i+1) , len_data) , : , : ] for i in range(k)]
        all_y = [y[fold_number*i : min(fold_number*(i+1) , len_data)] for i in range(k)]
        print("\n" + "="*60)
        print(f"Subject {person_num}: Training {k}-fold cross-validation")
        print("="*60)
        for i in range(fold_start, k): 
            print(f"\n-- Fold {i+1}/{k} --")
            x_test = all_x[i]
            y_test = all_y[i]
            x_train = all_x[:i] + all_x[i+1:]
            y_train = all_y[:i] + all_y[i+1:]
            x_train = torch.concat(x_train , dim=0)
            y_train = torch.concat(y_train , dim=0)
            test_dataset = TensorDataset(x_test , y_test)
            test_loader = DataLoader(test_dataset ,batch_size , shuffle=False)
            train_dataset = TensorDataset(x_train , y_train )
            train_loader = DataLoader(train_dataset , batch_size,shuffle=True )
            Model = model (num_filter, 128* time_len, caps_len, num_emotions, out_dim)
            unique_Loss_fn = lambda v , y : loss_fn(v , y) # معماری دلخواه        
            subj_dir = os.path.join(run_dir or ".", f"subject_{person_num}")
            ensure_dir(subj_dir)
            fold_dir = os.path.join(subj_dir, f"fold_{i}")
            ensure_dir(fold_dir)
            checkpoint_path = os.path.join(fold_dir, "checkpoint.pth")
            log_path = os.path.join(fold_dir, "log.json")
            status_path = os.path.join(fold_dir, "status.json")
            #____trainer_______#
            trainer = Trainer(
                model=Model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                label_method=category,
                optimizer_cls=torch.optim.Adam,
                lr=1e-4,
                epochs=30,
                loss_fn = unique_Loss_fn, 
                verbose=True,
                save_each_epoch=False,
                checkpoint_path=checkpoint_path,
                log_path=log_path,
                status_path=status_path,
            )
            #____fit_model_____#
            history =  trainer.fit()
            fold_train_acc = np.mean(np.array(history['train_acc'][-5:]))
            fold_val_acc = np.mean(np.array(history['val_acc'][-5:]))
            print(f"Fold {i+1} result -> Train Acc (last5 avg): {fold_train_acc:.2f}% | Test Acc (last5 avg): {fold_val_acc:.2f}%")
            if i == fold_start: 
                train_loss = np.array(history['train_loss'])
                val_loss = np.array(history['val_loss'])
                train_acc = np.array(history['train_acc'])
                val_acc = np.array(history['val_acc'])
            else : 
                train_loss += np.array(history['train_loss'])
                val_loss += np.array(history['val_loss'])
                train_acc += np.array(history['train_acc'])
                val_acc += np.array(history['val_acc'])
            status.update({
                "current_subject": person_num,
                "current_fold": i + 1,
                "total_subjects": total_subjects,
                "total_folds": k,
                "status": "running",
            })
            save_json(status_file, status)
        train_acc  /=k
        train_loss /=k
        val_loss   /=k
        val_acc    /=k

        final_train_acc = np.mean(np.array(train_acc[-5:]))
        final_test_acc = np.mean(np.array(val_acc[-5:]))
        accuracies_on_subjects['train'].append(final_train_acc)
        accuracies_on_subjects['test'].append(final_test_acc)
        
        # ذخیره نتایج این سوژه در پوشه subject
        subj_dir = os.path.join(run_dir or ".", f"subject_{person_num}")
        ensure_dir(subj_dir)
        subj_status_file = os.path.join(subj_dir, "status.json")
        save_json(subj_status_file, {
            "train_acc": final_train_acc,
            "test_acc": final_test_acc,
            "completed": True
        })
        
        # ذخیره history کلی
        save_json(history_file, accuracies_on_subjects)
        
        status.update({
            "current_subject": person_num + 1,
            "current_fold": 0,
            "status": "running",
        })
        save_json(status_file, status)
    status.update({"status": "completed"})
    save_json(status_file, status)
    return accuracies_on_subjects












