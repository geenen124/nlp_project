from __future__ import print_function
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
import json
import h5py


class EasyDataset(Dataset):
    def __init__(self, data_directory, training_file,
                 image_mapping_file, image_feature_file):
        self.image_features = np.asarray(
                h5py.File(
                    data_directory + image_feature_file,
                    'r')['img_features'])

        with open(data_directory + image_mapping_file, 'r') as f:
            self.image_id_feature_mapping = json.load(f)['IR_imgid2id']

        with open(data_directory + training_file, 'r') as f:
            self.training_data = list(json.load(f).values())

        self.vocabulary_set = self.load_vocab_set()

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, item_index):
        training_item_dict = self.training_data[item_index].copy()
        image_features = dict()
        for image_index, image_id in enumerate(training_item_dict['img_list']):
            this_images_features = self.image_features[
                    self.image_id_feature_mapping[str(image_id)]]
            image_features[image_id] = this_images_features
            # Also add this as the target image features if it corresponds
            if image_index == training_item_dict['target']:
                training_item_dict['target_img_features'] = \
                        this_images_features

        training_item_dict['img_features'] = image_features

        return training_item_dict

    def get_vocab_set(self):
        vocab = set()
        for word_dict in self.dataset.training_data:
            caption = word_dict["caption"].lower().split(" ")
            vocab = vocab.union(caption)
        return vocab

    def save_vocab_set_to_file(self, csv_name="vocab_set"):
        file_name = "./data/"+csv_name+".csv"
        with open(file_name, "w") as vocab_file:
            wr = csv.writer(vocab_file, delimiter='|', quoting=csv.QUOTE_ALL)
            wr.writerow(self.vocabulary_set)

    def load_vocab_set(self, csv_name="vocab_set"):
        vocab = set()
        file_name = "./data/"+csv_name+".csv"
        with open(file_name, "r") as vocab_file:
            reader = csv.reader(vocab_file, delimiter='|', quoting=csv.QUOTE_ALL)
            for row in reader:
                vocab = vocab.union(row)
        return vocab
