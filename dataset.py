import os
from glob import glob
import pandas as pd

from torch import tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

class GetDataset(Dataset):
    def __init__(self, task='CoLA', phase='train', checkpoint='skt/kogpt2-base-v2'):
        super().__init__()
        self.task = task
        self.phase = phase
        self.data = self.get_data(task, phase)
        self.tokenizer = self.get_tokenizer(checkpoint)
        self.encoded, self.labels = self.process()

    def get_tokenizer(self, checkpoint):
        return PreTrainedTokenizerFast.from_pretrained(checkpoint, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>') 

    def process(self):
        self.data['sentence'] = self.data.apply(lambda x: '<s>' + x['sentence'][:-1] + '</s>', axis = 1)
        sentence = self.data['sentence'].values.tolist()
        labels = self.data['acceptability_label'].values.tolist()
        return self.tokenizer(sentence, truncation=True, padding=True), labels

    def get_data(self, task, phase):
        if task == 'CoLA':
            source = 'NIKL'
            if phase == 'validation':
                phase = 'dev'
            return pd.read_csv(f'/opt/ml/input/ck/{source}_{task}_{phase}.tsv', sep = '\t')
        else:
            if phase == 'validation':
                phase = 'Dev'
            elif phase == 'train':
                phase = 'Train'
            elif phase == 'test':
                phase = 'Test'
        
            if task == 'WiC':
                source = 'NIKL_SKT'
                return pd.read_csv(f'/opt/ml/input/ck/{source}_{task}_{phase}.tsv', sep = '\t')
            elif task == 'BoolQ' or task == 'COPA':
                source = 'SKT'
                return pd.read_csv(f'/opt/ml/input/ck/{source}_{task}_{phase}.tsv', sep = '\t')

    def __getitem__(self, index):
        item = {key: tensor(val[index]) for key, val in self.encoded.items()}
        item['labels'] = tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)
