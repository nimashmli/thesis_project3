from dataset.main import data , data_for_subject_dependet
import torch 
import os # os را برای چک کردن cuda اضافه کنید
from models_structures.hippoLegS1 import model
from train import Trainer
import random
from functions import k_fold_data_segmentation
from  torch.utils.data import DataLoader , TensorDataset
import numpy as np 
import torch.nn as nn
#____Model______#
def create_model(test_person , emotion,category , fold_idx, run_dir=None, config_path=None, config=None, resume=False) : 
    from pathlib import Path
    
    overlap = 0
    time_len = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if category == 'binary'  :
        output_dim = 2 
    elif category == '5category' :
        output_dim = 5
    batch_size = 64
    data_type = torch.float32
    my_dataset = data(test_person, overlap, time_len, device, emotion, category, batch_size, data_type)
    train_loader = my_dataset.train_data()
    test_loader = my_dataset.test_data()

    x_dim , h_dim , seq_len ,c_dim = 14 , 24, 128*time_len, 64
    dim2 , dim3  = 64 , 16 
    Model = model( x_dim, h_dim, c_dim   ,seq_len,dim2 , dim3 , output_dim)# معماری دلخواه
    # class weights for imbalance
    y_train = my_dataset.y_train
    class_count = torch.bincount(y_train.long())
    class_count = class_count + (class_count == 0).long()
    weights = (1.0 / class_count.float())
    weights = weights / weights.sum() * len(class_count)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # تعیین مسیر checkpoint و log
    if run_dir:
        run_dir = Path(run_dir)
        checkpoint_path = run_dir / f"checkpoint_fold{fold_idx}.pth"
        log_path = run_dir / f"log_fold{fold_idx}.json"
    else:
        checkpoint_path = f"eeg_checkpoint{fold_idx }.pth"
        log_path = f"eeg_log{fold_idx }.json"

    #____trainer_______#
    trainer = Trainer(
        model=Model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        label_method=category,
        optimizer_cls=torch.optim.Adam,
        lr=5e-5,
        epochs=25,
        loss_fn = criterion ,
        checkpoint_path=str(checkpoint_path),
        log_path=str(log_path),
        config_path=config_path,
        save_each_epoch=True
    )
    #____fit_model_____#
    return  trainer.fit()

def subject_dependent_validation (emotion ,category, fold_idx , k=5, run_dir=None, config_path=None, config=None, resume=False) : 
    from pathlib import Path
    from experiment_manager import ExperimentManager
    
    overlap = 0
    time_len = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if category == 'binary'  :
        output_dim = 2 
    elif category == '5category' :
        output_dim = 5
    batch_size = 64
    data_type = torch.float32
    
    # بارگذاری نتایج قبلی اگر resume=True باشد
    accuracies_on_subjects = {
        'train' : [] , 
        'test' : []
    }
    start_subject = 0
    start_fold = 0
    current_subject = -1
    
    if resume and config and config_path:
        manager = ExperimentManager()
        # خواندن آخرین subject و fold پردازش شده
        last_subject = config.get('last_completed_subject', -1)
        last_fold = config.get('last_completed_fold', -1)
        current_subject = config.get('current_subject', -1)
        
        # اگر subject قبلی کامل شده، از subject بعدی شروع کن
        if last_fold == k - 1:  # همه foldهای subject قبلی کامل شده
            start_subject = last_subject + 1
            start_fold = 0
        else:  # subject قبلی نیمه‌کاره است
            start_subject = current_subject
            start_fold = last_fold + 1
        
        # خواندن نتایج قبلی
        if 'accuracies' in config:
            accuracies_on_subjects['train'] = config['accuracies'].get('train', [])
            accuracies_on_subjects['test'] = config['accuracies'].get('test', [])
        
        print(f"\n🔄 Resuming from Subject {start_subject}, Fold {start_fold + 1} (previous subjects: {len(accuracies_on_subjects['train'])} completed)")
    
    person_num = start_subject
    data = data_for_subject_dependet(overlap , time_len ,emotion ,category ,data_type , device , k  )
    
    # تبدیل iterator به لیست برای امکان skip کردن
    data_list = list(data.data)
    
    # شروع از subject مشخص شده
    for subject_idx, (x , y) in enumerate(data_list[start_subject:], start=start_subject): 
        # به‌روزرسانی current_subject در config
        if config_path and run_dir:
            manager = ExperimentManager()
            manager.update_experiment_config(
                config_path,
                current_subject=person_num
            )
        
        # اگر subject جدید است، از fold 0 شروع کن، وگرنه از fold مشخص شده
        if subject_idx == start_subject:
            fold_start = start_fold
        else:
            fold_start = 0
        
        fold_idx = fold_start
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
            x_dim , h_dim , seq_len ,c_dim = 14 , 32 , 128*time_len, 32
            dim2 , dim3  = 64 , 16
            Model = model( x_dim, h_dim, c_dim   ,seq_len,dim2 , dim3 , output_dim)# معماری دلخواه
            criterion = nn.CrossEntropyLoss()

            # تعیین مسیر checkpoint و log
            if run_dir:
                run_dir = Path(run_dir)
                subject_dir = run_dir / f"subject_{person_num}"
                subject_dir.mkdir(exist_ok=True)
                checkpoint_path = subject_dir / f"checkpoint_fold{i}.pth"
                log_path = subject_dir / f"log_fold{i}.json"
            else:
                checkpoint_path = f"eeg_checkpoint{fold_idx + person_num*5}.pth"
                log_path = f"eeg_log{fold_idx + person_num*5}.json"

            #____trainer_______#
            # برای subject_dependent نیازی به ذخیره مدل نیست، فقط log کافی است
            trainer = Trainer(
                model=Model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                label_method=category,
                optimizer_cls=torch.optim.Adam,
                lr=5e-5,
                epochs=30,
                loss_fn = criterion ,
                verbose=True,
                save_each_epoch=False,  # نیازی به ذخیره مدل نیست
                checkpoint_path=str(checkpoint_path),
                log_path=str(log_path),
                config_path=None  # برای subject_dependent نیازی به به‌روزرسانی config در trainer نیست
            )
            #____fit_model_____#
            history =  trainer.fit()
            fold_train_acc = np.mean(np.array(history['train_acc'][-5:]))
            fold_val_acc = np.mean(np.array(history['val_acc'][-5:]))
            print(f"Fold {i+1} result -> Train Acc (last5 avg): {fold_train_acc:.2f}% | Test Acc (last5 avg): {fold_val_acc:.2f}%")
            if fold_idx ==0 : 
                train_loss = np.array(history['train_loss'])
                val_loss = np.array(history['val_loss'])
                train_acc = np.array(history['train_acc'])
                val_acc = np.array(history['val_acc'])
            else : 
                train_loss += np.array(history['train_loss'])
                val_loss += np.array(history['val_loss'])
                train_acc += np.array(history['train_acc'])
                val_acc += np.array(history['val_acc'])
            # به‌روزرسانی config بعد از هر fold
            if config_path and run_dir:
                manager = ExperimentManager()
                manager.update_experiment_config(
                    config_path,
                    last_completed_fold=i,
                    current_subject=person_num
                )
            
            fold_idx +=1
        
        # بعد از کامل شدن همه foldهای یک subject
        train_acc  /=k
        train_loss /=k
        val_loss   /=k
        val_acc    /=k

        accuracies_on_subjects['train'].append(np.mean(np.array(train_acc[-5:])))
        accuracies_on_subjects['test'].append(np.mean(np.array(val_acc[-5:])))
        
        # به‌روزرسانی config بعد از کامل شدن subject
        if config_path and run_dir:
            manager = ExperimentManager()
            manager.update_experiment_config(
                config_path,
                last_completed_subject=person_num,
                last_completed_fold=k-1,  # همه foldها کامل شده
                accuracies={
                    'train': accuracies_on_subjects['train'],
                    'test': accuracies_on_subjects['test']
                }
            )
            print(f"✅ Subject {person_num} completed and saved to config")
        
        person_num +=1
    
    return accuracies_on_subjects













