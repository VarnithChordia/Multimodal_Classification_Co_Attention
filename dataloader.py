from torch.utils.data import Dataset
import torch
from PIL import Image
import os
from torch.utils.data import DataLoader
from transformers import CamembertTokenizer, FlaubertTokenizer, XLMRobertaTokenizer, BertTokenizer
import json
import pickle



class InputFeature(object):
    """
    A single set of features
    """

    def __init__(self, labels, input_ids, input_mask, input_type_ids, image):
        self.data = []
        self.data.append(input_ids)
        self.data.append(input_mask)
        self.data.append(labels)
        self.data.append(image)
        self.data.append(input_type_ids)

    def __iter__(self):
        return iter(self.data)


class TextLoader(DataLoader):
    def __init__(self, data_set, shuffle=False, device="cuda", batch_size=16,num_workers=2):
        super(TextLoader, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers = num_workers
        )
        self.device = device

    def collate_fn(self, data):
        res = []
        token_ml = max(map(lambda x_: sum(x_.data[1]), data))
        for sample in data:
            example = []
            for x in sample:
                if isinstance(x, list):
                    x = x[:token_ml]
                example.append(x)
            res.append(example)
        res_ = []
        for idx, x in enumerate(zip(*res)):
            if  isinstance(x[0], list):
                res_.append(torch.LongTensor(x))
            elif isinstance(x[0], str):
                res_.append(torch.LongTensor([int(values) for values in x]))
            else:
                res_.append(torch.stack([value for value in x]))
        # return [t.to(self.device) for t in res_]
        return {'image':res_[3], 'input_id':res_[0], 'input_mask': res_[1], 'labels':res_[2] ,'input_type_ids':res_[4]}



class VQADataset(Dataset):
    """VQA Dataset"""

    def __init__(self, data, img_dir, transform,tokenizer,config):
        """

        :param data:
        :param img_dir:
        :param transform:
        :param tokenizer:
        :param config:
        """
        self.data = data
        self.images_dir = img_dir
        # Image transforms
        self.transform = transform
        self.config = config
        self.tokenizer = tokenizer



    @classmethod
    def create(cls, data_file,
               image_dir,
               transform,
               labels_path,
               pad_idx =0,
               tokenizer =  None,
               model_type = None,
               min_char_len = 1,
               max_seq_length=510,
               model_name = "camembert-base",
               clear_cache = False,
               is_cls = True):
        if tokenizer is None:
            if 'camem' in model_type:
                tokenizer = CamembertTokenizer.from_pretrained(model_name)
            elif 'flaubert' in model_type:
                tokenizer = FlaubertTokenizer.from_pretrained(model_name)
            elif 'XLMRoberta' in model_type:
                tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
            elif 'M-Bert' in model_type:
                tokenizer = BertTokenizer.from_pretrained(model_name)

        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        # data =  data_file

        idx2labels, labels2idx = cls.create_labels(labels_path)
        config = {
            "min_char_len": min_char_len,
            "model_name": model_name,
            "max_sequence_length": max_seq_length,
            "clear_cache": clear_cache,
            "pad_idx": pad_idx,
            "is_cls": is_cls,
            "idx2labels":idx2labels,
            "labels2idx":labels2idx
        }


        self = cls(data,image_dir,transform,tokenizer,config)

        return self

    @classmethod
    def create_labels(cls,labels_path):
        with open(labels_path,'r') as f:
            idx2labels = json.load(f)
        labels2idx = dict((values,keys) for (keys,values) in idx2labels.items())

        return idx2labels, labels2idx



    def create_features(self,item):
        # bert_tokens = []
        # tok_map = []
        img_path = os.path.join(self.images_dir, item[1])
        cur_tokens = self.tokenizer.tokenize(item[2][:self.config["max_sequence_length"]])
        cur_label = item[3]
        cur_label = self.config['labels2idx'][cur_label]
        orig_tokens = cur_tokens
        input_ids = self.tokenizer.encode(orig_tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.config["max_sequence_length"]:
            input_ids.append(self.config["pad_idx"])
            input_mask.append(0)

        input_type_ids = [0] * len(input_ids)

        # img_path = os.path.join(self.images_dir, item[1])
        image = Image.open(img_path).convert('RGB')


        image = self.transform(image)


        return InputFeature(
            input_ids=input_ids,
            input_mask=input_mask,
            labels=cur_label,
            image=image,
            input_type_ids=input_type_ids
        )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Parse the text file line
        data_item = self.data[idx]

        return self.create_features(data_item)
