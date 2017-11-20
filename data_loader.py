from __future__ import print_function
import os
import torch
import numpy as np
import string
from torch.utils.data import Dataset, DataLoader
import csv
import json
import h5py
from collections import Counter, defaultdict


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

        self.w2i = {}
        self.i2w = {}
        self.vocab_size = 0

        #populate w2i, i2w, and get vocab size after simple preprocessing
        self.get_vocab_set()


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
        print ("Loading Vocabulary")
        vocab_counter = Counter()
        punctuation_remover = str.maketrans('', '', string.punctuation)
        min_count = 3

        for word_dict in self.training_data:
            #Preprocessing
            caption = word_dict["caption"].translate(punctuation_remover).lower().split(" ")
            vocab_counter.update(caption)

        vocabulary_set = set()
        unk_added = False
        for word in set(list(vocab_counter.elements())):
            if vocab_counter[word] > min_count and word:
                vocabulary_set.add(word)
            #tag unknown token (UNK) the words with less than 5 or 3 counts
            elif not unk_added:
                vocabulary_set.add("UNK")
                unk_added = True

        #Construct w2i and i2w
        self.vocab_size = len(vocabulary_set)

        iterable_vocab = list(vocabulary_set)
        iterable_vocab.sort()

        for index, word in enumerate(iterable_vocab):
            self.w2i[word] = index
            self.i2w[index] = word

        #make defaultdit that points value to index of unknown tag UNK
        self.w2i = defaultdict(lambda: self.w2i["UNK"], self.w2i)
