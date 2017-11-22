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
import spacy
import pickle
nlp = spacy.load('en')

#Dataset Class - combines json and image into a training example
class SimpleDataset(Dataset):
    def __init__(self, data_directory, training_file,
                 image_mapping_file, image_feature_file,
                 preprocessing=False, preprocessed_data_filename="easy_dataset_preprocessed"):
        self.image_features = np.asarray(
                h5py.File(
                    data_directory + image_feature_file,
                    'r')['img_features'])

        with open(data_directory + image_mapping_file, 'r') as f:
            self.image_id_feature_mapping = json.load(f)['IR_imgid2id']

        with open(data_directory + training_file, 'r') as f:
            self.training_data = list(json.load(f).values())

        self.preprocessed_data_filename = preprocessed_data_filename

        if os.path.isfile("data/" + self.preprocessed_data_filename+".pkl"):
            self.load_preprocessed_dict()

        else:
            # self.training_data_processed = {}
            self.w2i = {}
            self.i2w = {}
            self.vocab_size = 0
            self.preprocessing = preprocessing

            #populate w2i, i2w, and get vocab size after simple preprocessing
            self.get_vocab_counter()

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

    #basic parsing of caption - used in non preprocessing
    def simple__caption_parsing(self, caption_str):
        punctuation_remover = str.maketrans('', '', string.punctuation)
        return caption_str.translate(punctuation_remover).lower().split(" ")

    #Using spacy and manual
    def get_vocab_counter(self):
        print ("Preprocessing Examples")
        vocab_counter = Counter()
        p_count = 0
        for word_dict in self.training_data:
            #Preprocessing if need be
            word_inputs = self.preprocess_example(word_dict)
            word_dict["processed_word_inputs"] = word_inputs

            vocab_counter.update(word_inputs)

            p_count+=1
            print(f"Preprocessed Count: {p_count}", end="\r")

        print("\n")
        self.generate_w2i_dict(vocab_counter)

    #returns a list of words to be kept/used
    def preprocess_example(self, word_dict):
        if not self.preprocessing:
            return self.simple__caption_parsing(word_dict["caption"])

        caption = self.preprocess_caption(word_dict["caption"])
        caption.extend(self.preprocess_questions(word_dict["dialog"]))

        return caption

    def preprocess_caption(self, caption_str):
        caption = []
        doc = nlp(caption_str)
        for token in doc:
            if not token.is_stop and not token.is_punct and token.text != "yes":
                caption.append(token.lemma_)
        return caption

    def preprocess_questions(self, question_list):
        question_text = []
        for question in question_list:
            answer = question[0].split(" ")[-1]
            if answer == "yes":
                question_text.extend(self.preprocess_caption(question[0]))
            elif answer != "no":
                question_text.append(answer)
        return question_text

    #using a min count, remove the least likely word from Counter,
    #and add the remaining words to w2i, i2w dictionaries
    def generate_w2i_dict(self, vocab_counter, min_count=3):
        print("Generating Dictionary")
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


        self.save_preprocessed_dict()
        #make defaultdit that points value to index of unknown tag UNK
        self.w2i = defaultdict(lambda: self.w2i["UNK"], self.w2i)


    def save_preprocessed_dict(self):
        print("Saving preprocessed data")
        master_dict = {}
        master_dict["w2i"] = self.w2i
        master_dict["i2w"] = self.i2w
        master_dict["vocab_size"] = self.vocab_size
        master_dict["preprocessing"] = self.preprocessing
        master_dict["training_data_processed"] = self.training_data
        with open('data/'+self.preprocessed_data_filename+'.pkl', 'wb') as f:
            pickle.dump(master_dict, f, pickle.HIGHEST_PROTOCOL)

    def load_preprocessed_dict(self):
        print("Loading preprocessed data")
        with open('data/' + self.preprocessed_data_filename + '.pkl', 'rb') as f:
            master_dict = pickle.load(f)
        self.w2i = master_dict["w2i"]
        self.w2i = defaultdict(lambda: self.w2i["UNK"], self.w2i)

        self.i2w = master_dict["i2w"]
        self.vocab_size = master_dict["vocab_size"]
        self.preprocessing = master_dict["preprocessing"]
        self.training_data = master_dict["training_data_processed"]
