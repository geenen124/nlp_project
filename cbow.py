from data_loader import EasyDataset
import csv
import torch
import torch.nn as nn
import torch.autograd as Variable
import torch.optim as optim
import torch.nn.functional as F


def dataset_to_index_dict(dataset):
    vocab_size = len(dataset.vocabulary_set)
    vocab_list = list(dataset.vocabulary_set)
    vocab_list.sort()
    word_2_index = {}
    for index, word in enumerate(vocab_list):
        word_2_index[word] = index
    return word_to_index


class CBOW(nn.Module):
    def __init__(self, vocab_size, image_feature_size, hidden_layer_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_layer_dim)
        self.hidden_layer = nn.Linear(
                vocab_size+image_feature_size,
                hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, 1)

    def forward(self, word_inputs, image_inputs):
        embeddings = self.embeddings(word_inputs)
        bow = torch.sum(embeddings, 1)
        # NB Image Inputs need to be a torch tensor 
        input_hidden = torch.cat((bow, image_inputs))
        output_hidden = self.hidden_layer(input_hidden)
        probability = F.sigmoid(self.output_layer(output_hidden))
        return probability


def train_network(dataset, num_epochs=1000, batch_size=64):

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True)
    for ITER in range(num_epochs):
        for sample_batch in dataloader:
            print(sample_batch)



if __name__ == '__main__':

    easy_dataset = EasyDataset(
            data_directory="./data/",
            training_file="IR_train_easy.json",
            image_mapping_file="IR_image_features2id.json",
            image_feature_file="IR_image_features.h5",
            )
    train_network(easy_dataset, num_epochs=1)
