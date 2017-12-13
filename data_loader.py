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
WORD_2_VEC_SIZE = 384

#Dataset Class - combines json and image into a training example
#preprocessing_type = stop_and_stem or word2vec

class SimpleDataset(Dataset):
    def __init__(self, training_file, preprocessing=False,
                 preprocessed_data_filename="easy_dataset_preprocessed",
                 preprocessing_type="stop_and_stem"):
        data_directory = "./data/"
        image_mapping_file = "IR_image_features2id.json"
        image_feature_file = "IR_image_features.h5"

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
            self.preprocessing_type = preprocessing_type

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

            if self.preprocessing_type != "w2v":
                vocab_counter.update(word_inputs)

            p_count+=1
            print(f"Preprocessed Count: {p_count}", end="\r")

        print("\n")
        self.generate_w2i_dict(vocab_counter)

    #returns a list of words to be kept/used
    def preprocess_example(self, word_dict):
        if not self.preprocessing:
            return self.simple__caption_parsing(word_dict["caption"])

        if self.preprocessing_type == "stop_and_stem":
            caption = self.preprocess_caption(word_dict["caption"])
            caption.extend(self.preprocess_questions(word_dict["dialog"]))
            return caption

        elif self.preprocessing_type == "w2v":
            word_vec = self.preprocess_caption(word_dict["caption"])
            question_vec, avg_count = self.preprocess_questions(word_dict["dialog"])
            if avg_count > 0: #average the question vector with the question vector
                word_vec = (question_vec + word_vec)/(avg_count+1)
            return word_vec

        else:
            print("Unknown preprocessing TYPE for Example")
            return self.simple__caption_parsing(word_dict["caption"])


    def preprocess_caption(self, caption_str):
        caption = []
        doc = nlp(caption_str)

        if self.preprocessing_type == "stop_and_stem":
            for token in doc:
                if not token.is_stop and not token.is_punct and token.text != "yes":
                    caption.append(token.lemma_)
            return caption

        elif self.preprocessing_type == "w2v":
            return doc.vector/doc.vector_norm

        else:
            print("Unknown preprocessing TYPE for Caption")
            return simple__caption_parsing(caption_str)


    def preprocess_questions(self, question_list):


        if self.preprocessing_type == "stop_and_stem":
            question_text = []
            for question in question_list:
                answer = question[0].split(" ")[-1]
                if answer == "yes":
                    question_text.extend(self.preprocess_caption(question[0]))
                elif answer != "no":
                    question_text.extend(self.preprocess_caption(question[0]))
                # elif answer != "no":
                    # question_text.extend(self.preprocess_caption(question[0].split("?")[-1]))
            return question_text

        elif self.preprocessing_type == "w2v":
            question_vec = np.zeros(WORD_2_VEC_SIZE)
            # avg_count = 0
            for question in question_list:
                answer = question[0].split(" ")[-1]
                if answer == "yes":
                    question_vec = question_vec+self.preprocess_caption(question[0])
                    # avg_count += 1
                elif answer == "no":
                    question_vec = question_vec-self.preprocess_caption(question[0])
                    # avg_count += 1
                else: #if answer is a word or sentence
                    question_vec = question_vec+self.preprocess_caption(question[0].replace("?", ""))
                    #  question_vec = question_vec+self.preprocess_caption(question[0].split("?")[-1])

            return question_vec, len(question_list)

        else:
            print("Unknown Preprocessing TYPE for Questions")
            return []

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
        self.vocab_size = len(vocabulary_set)+1

        iterable_vocab = list(vocabulary_set)
        iterable_vocab.sort()

        self.w2i["PAD"] = 0
        self.i2w[0] = "PAD"

        for index, word in enumerate(iterable_vocab):
            self.w2i[word] = index+1
            self.i2w[index+1] = word

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
        master_dict["preprocessing_type"] = self.preprocessing_type
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
        self.preprocessing_type = master_dict["preprocessing_type"]
        self.training_data = master_dict["training_data_processed"]
