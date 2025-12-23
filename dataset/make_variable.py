import torch 
import os 
import pickle as pik
import torch.nn as nn 

def extract_and_tensor (path , dtype) : 
    if os.path.exists(path) : 
        with open(path , 'rb' ) as f : 
            data = pik.load(f)
        torch_data = torch.tensor(data , dtype = dtype  )
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
    def __init__(self , test_person  , overlap  , time_lenght    , emotion , label_method  ) :
        super().__init__()
        self.test_person = test_person
        self.train_person = list(range(23))
        for i in self.test_person : 
            self.train_person.remove(i)
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

    def extract (self , file_path , dtype ) : 
        base_extracted_dir = file_path['base_extracted_dir']
        label_file_path = file_path['labels_file']
        stimuli_files = file_path['stimuli_files']
        baseline_data = extract_and_tensor(base_extracted_dir, dtype) #(23, 18 , 7808, 14)
        labels = extract_and_tensor(label_file_path, dtype)

        all_train_data_slices = []
        all_train_label_slices = []
        all_test_data_slices = []
        all_test_label_slices = []

        for i in range(18):
            stimuli_data = extract_and_tensor(stimuli_files[i], dtype=dtype)

            for j in self.train_person:
                person_stimuli = stimuli_data[j, :, :]
                sliced_stimuli = slice_data(person_stimuli, self.overlap, self.time_lenght)
                sliced_baseline= slice_data(baseline_data[j][i] , 0 , self.time_lenght)
                sliced_baseline = torch.sum(sliced_baseline , dim=0)/sliced_baseline.shape[0]
                sliced_stimuli -= sliced_baseline
                if sliced_stimuli is not None and sliced_stimuli.shape[0] > 0:
                    all_train_data_slices.append(sliced_stimuli)
                    
                    num_slices = sliced_stimuli.shape[0]
                    current_label = labels[j][i][self.emotion_number]
                    sliced_labels = torch.full((num_slices,), current_label, dtype=torch.long)
                    all_train_label_slices.append(sliced_labels)

            # حلقه روی افراد آزمایشی
            for j in self.test_person:
                person_stimuli = stimuli_data[j, :, :]
                sliced_stimuli = slice_data(person_stimuli, self.overlap, self.time_lenght)
                sliced_baseline= slice_data(baseline_data[j][i] , 0 , self.time_lenght)
                sliced_baseline = torch.sum(sliced_baseline , dim=0)/sliced_baseline.shape[0]
                sliced_stimuli -= sliced_baseline
                if sliced_stimuli is not None and sliced_stimuli.shape[0] > 0:
                    all_test_data_slices.append(sliced_stimuli)
                    
                    num_slices = sliced_stimuli.shape[0]
                    current_label = labels[j][i][self.emotion_number]
                    sliced_labels = torch.full((num_slices,), current_label, dtype=torch.long)
                    all_test_label_slices.append(sliced_labels)

        # concat کردن تمام تنسورها در بعد صفرم (dim=0) و اختصاص به متغیرهای کلاس
        self.train_data = torch.cat(all_train_data_slices, dim=0)
        self.train_labels = torch.cat(all_train_label_slices, dim=0)
        self.test_data = torch.cat(all_test_data_slices, dim=0)
        self.test_labels = torch.cat(all_test_label_slices, dim=0)
        if self.label_method == 'binary' : 
            self.train_labels = (self.train_labels > 2).long() 
            self.test_labels = (self.test_labels > 2).long()
        else : 
            self.train_labels -= 1 
            self.test_labels -=1 
            self.train_labels = self.train_labels.long()
            self.test_labels = self.test_labels.long()



    def normalize(self):
            self.mean = self.train_data.mean()
            self.variance = self.train_data.var()
            if self.variance == 0:
                self.variance = torch.tensor(1e-4, device=self.device)
            std_dev = torch.sqrt(self.variance)
            self.train_data = (self.train_data - self.mean)/ std_dev
            self.test_data = (self.test_data- self.mean) / std_dev
    def recieve_data(self) : 
        return self.train_data , self.test_data , self.train_labels , self.test_labels


