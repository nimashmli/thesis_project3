import torch 
import os 
import pickle as pik
import torch.nn as nn 
import random
from torch.utils.data  import Dataset , DataLoader
import numpy as np

def extract_and_tensor(path, dtype):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pik.load(f)
        if isinstance(data, np.ndarray):
            torch_data = torch.from_numpy(data).to(dtype)
        else:
            torch_data = torch.tensor(data, dtype=dtype)
        return torch_data

def slice_data(data, overlap, time_len):
    sampling_rate = 128
    slice_len = int(time_len * sampling_rate)
    #step = int(slice_len * (1 - overlap))
    overlap_samples = int(slice_len * (overlap / 100))
    step = slice_len - overlap_samples
    sliced_data = []
    for start_index in range(0, data.shape[0] - slice_len + 1, step):
        data_slice = data[start_index : start_index + slice_len, :]
        sliced_data.append(data_slice)
    if sliced_data:
        return torch.stack(sliced_data)
    else:
        return torch.empty(0, slice_len, data.shape[1], dtype=data.dtype)

class dataset(nn.Module) : 
    def __init__(self  , overlap  , time_lenght    , emotion , label_method  ) :
        super().__init__()
        self.overlap = overlap
        self.time_lenght = time_lenght
        self.emotion = emotion
        self.emotion_number = 0
        if self.emotion == 'valence' : 
            self.emotion_number  =1
        elif self.emotion=="dominance" : 
            self.emotion_number = 2
        self.label_method = label_method
        self.variance =  1
        self.mean = 0

    def extract (self , file_path , dtype, person , test_clips ) : 
        base_extracted_dir = file_path['base_extracted_dir']
        label_file_path = file_path['labels_file']
        stimuli_files = file_path['stimuli_files']
        baseline_data = extract_and_tensor(base_extracted_dir, dtype) #(23, 18 , 7808, 14)
        labels = extract_and_tensor(label_file_path, dtype)

        train_clips = list(range(18))
        for test_clip_num in test_clips : 
            train_clips.remove(test_clip_num)
        train_data , train_labels = self.make_data(train_clips, person , dtype, stimuli_files, baseline_data ,labels  )
        test_data , test_labels = self.make_data(test_clips, person , dtype, stimuli_files, baseline_data ,labels  )

        mean = train_data.mean()
        var  = train_data.var()
        if var < 1e-4: 
            var = torch.tensor(1e-4 )
        std = torch.sqrt(var)
        train_data = (train_data - mean)/(std)
        test_data = (test_data  - mean)/(std)
        return train_data , train_labels , test_data , test_labels 


    def make_data(self , clip_numbers, person , dtype, stimuli_files, baseline_data ,labels  ) : 
        all_train_data_slices = []
        all_train_label_slices = []
        for i in clip_numbers:
            stimuli_data = extract_and_tensor(stimuli_files[i], dtype=dtype)
            person_stimuli = stimuli_data[person, :, :]
            sliced_stimuli = slice_data(person_stimuli, self.overlap, self.time_lenght)
            sliced_baseline= slice_data(baseline_data[person][i] , 0 , self.time_lenght)
            sliced_baseline = torch.sum(sliced_baseline , dim=0)/sliced_baseline.shape[0]
            sliced_stimuli -= sliced_baseline
            if sliced_stimuli is not None and sliced_stimuli.shape[0] > 0:
                all_train_data_slices.append(sliced_stimuli)
                
                num_slices = sliced_stimuli.shape[0]
                current_label = labels[person][i][self.emotion_number]
                sliced_labels = torch.full((num_slices,), current_label, dtype=torch.long)
                all_train_label_slices.append(sliced_labels)

        train_data = torch.cat(all_train_data_slices, dim=0) # (batch , time_len , 14)
        train_labels = torch.cat(all_train_label_slices, dim=0) #(batch,)
        if self.label_method == 'binary' : 
            train_labels = (train_labels > 2).long() 
        else : 
            train_labels -= 1 
            train_labels= train_labels.long()
        return train_data , train_labels



class shuffled_dataset : 
    def __init__(self, dtype , file_path , label_method,overlap,time_lenght, emotion , device) : 
        self.data = [] 
        self.overlap = overlap
        self.time_lenght = time_lenght
        self.emotion = emotion
        self.emotion_number = 0
        if self.emotion == 'valence' : 
            self.emotion_number  =1
        elif self.emotion=="dominance" : 
            self.emotion_number = 2
        self.label_method = label_method
        self.variance =  1
        self.mean = 0
        clip_numbers = list(range(18))
        base_extracted_dir = file_path['base_extracted_dir']
        label_file_path = file_path['labels_file']
        stimuli_files = file_path['stimuli_files']
        baseline_data = extract_and_tensor(base_extracted_dir, dtype) #(23, 18 , 7808, 14)
        labels = extract_and_tensor(label_file_path, dtype)
        for person in range(23) :
            x , y = self.make_data(clip_numbers, person , dtype, stimuli_files, baseline_data ,labels  ) 
            idx =  torch.randperm(x.shape[0])
            x = x[idx , : ,: ]
            y  = y[idx]
            print(f'person  {person} have {y.shape}')
            self.data.append((x.to(device) , y.to(device) ))

    def make_data(self , clip_numbers, person , dtype, stimuli_files, baseline_data ,labels  ) : 
        all_train_data_slices = []
        all_train_label_slices = []
        for i in clip_numbers:
            stimuli_data = extract_and_tensor(stimuli_files[i], dtype=dtype)
            person_stimuli = stimuli_data[person, :, :]
            sliced_stimuli = slice_data(person_stimuli, self.overlap, self.time_lenght)
            sliced_baseline= slice_data(baseline_data[person][i] , 0 , self.time_lenght)
            sliced_baseline = torch.sum(sliced_baseline , dim=0)/sliced_baseline.shape[0]
            sliced_stimuli -= sliced_baseline
            if sliced_stimuli is not None and sliced_stimuli.shape[0] > 0:
                all_train_data_slices.append(sliced_stimuli)
                
                num_slices = sliced_stimuli.shape[0]
                current_label = labels[person][i][self.emotion_number]
                sliced_labels = torch.full((num_slices,), current_label, dtype=torch.long)
                all_train_label_slices.append(sliced_labels)

        train_data = torch.cat(all_train_data_slices, dim=0) # (batch , time_len , 14)
        train_labels = torch.cat(all_train_label_slices, dim=0) #(batch,)
        if self.label_method == 'binary' : 
            train_labels = (train_labels > 2).long() 
        else : 
            train_labels -= 1 
            train_labels= train_labels.long()
        return train_data , train_labels

class kfold_dataset:
    def __init__(self, dtype, file_path, label_method, overlap, time_len, emotion, device, k=5):
        self.data = []  # لیست خروجی (data, label) برای هر شخص
        self.overlap = overlap
        self.time_len = time_len
        self.emotion = emotion
        self.label_method = label_method
        self.device = device
        self.k = k
        self.emotion_number = 1 if emotion == 'valence' else 2 if emotion == 'dominance' else 0

        clip_numbers = list(range(18))
        base_extracted_dir = file_path['base_extracted_dir']
        label_file_path = file_path['labels_file']
        stimuli_files = file_path['stimuli_files']
        baseline_data = extract_and_tensor(base_extracted_dir, dtype)
        labels = extract_and_tensor(label_file_path, dtype)

        for person in range(23):
            person_data_slices = []
            person_label_slices = []

            # تقسیم هر کلیپ به k بخش
            for clip_idx in clip_numbers:
                stimuli_data = extract_and_tensor(stimuli_files[clip_idx], dtype=dtype)
                person_stimuli = stimuli_data[person, :, :]
                sliced_stimuli = slice_data(person_stimuli, overlap, time_len)

                baseline_clip = slice_data(baseline_data[person][clip_idx], 0, time_len)
                baseline_avg = torch.sum(baseline_clip, dim=0) / baseline_clip.shape[0]
                sliced_stimuli -= baseline_avg

                if sliced_stimuli.shape[0] > 0:
                    person_data_slices.append(sliced_stimuli)
                    num_slices = sliced_stimuli.shape[0]
                    current_label = labels[person][clip_idx][self.emotion_number]
                    sliced_labels = torch.full((num_slices,), current_label, dtype=torch.long)
                    person_label_slices.append(sliced_labels)

            # concat کلیپ‌ها برای شخص
            all_data = torch.cat(person_data_slices, dim=0)
            all_labels = torch.cat(person_label_slices, dim=0)

            # shuffle فقط داخل هر بخش k-fold
            fold_size = all_data.shape[0] // k
            folds_data = []
            folds_label = []
            for i in range(k):
                start = i * fold_size
                end = (i + 1) * fold_size if i < k-1 else all_data.shape[0]
                folds_data.append(all_data[start:end])
                folds_label.append(all_labels[start:end])

            # concat دوباره همه foldها
            final_data = torch.cat(folds_data, dim=0)
            final_label = torch.cat(folds_label, dim=0)

            # normalize per-person using statistics from this person's data only
            mean = final_data.mean()
            var = final_data.var()
            if var < 1e-4:
                var = torch.tensor(1e-4, dtype=final_data.dtype, device=final_data.device)
            std = torch.sqrt(var)
            final_data = (final_data - mean) / std

            # تنظیم برچسب‌ها برای CrossEntropy
            if label_method == 'binary':
                final_label = (final_label > 2).long()
            else:
                final_label = final_label - 1
                final_label = final_label.long()

            self.data.append((final_data.to(device), final_label.to(device)))
            print(f"person {person} have {final_label.shape}")
