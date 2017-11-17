from data_loader import EasyDataset
import csv
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


def dataset_to_index_dict_and_size(dataset):
    vocab_size = len(dataset.vocabulary_set)
    vocab_list = list(dataset.vocabulary_set)
    vocab_list.sort()
    word_2_index = {}
    for index, word in enumerate(vocab_list):
        word_2_index[word] = index
    return word_2_index, vocab_size


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_space, image_feature_size, hidden_layer_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_space)
        self.hidden_layer = nn.Linear(
                embedding_space+image_feature_size,
                hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, 1)

    def forward(self, word_inputs, image_inputs):
        embeddings = self.embeddings(word_inputs)
        bow = torch.sum(embeddings, 0)
        # NB Image Inputs need to be a torch tensor 
        input_hidden = torch.cat((bow, image_inputs))
        output_hidden = self.hidden_layer(input_hidden)
        probability = F.sigmoid(self.output_layer(output_hidden))
        return probability


def train_network(dataset, num_epochs=1000, batch_size=64):
    word_to_index, vocab_size = dataset_to_index_dict_and_size(dataset)
    image_feature_size = 2048
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda x: x,
            shuffle=True)

    # Actually make the model
    model = CBOW(vocab_size, 200, image_feature_size, hidden_layer_dim=256)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_loss = 0.0

    for ITER in range(num_epochs):
        print(f"Training Loss {train_loss}")
        train_loss = 0.0

        for sample_batch in dataloader:
            for sample in sample_batch:
                caption = sample["caption"].lower().split(" ")
                lookup_tensor = Variable(
                    torch.LongTensor([word_to_index[x] for x in caption])
                    )
                # Forward and backward pass per image, text is fixed
                for image_id in sample['img_list']:
                    image_features = sample['img_features'][image_id]
                    image_features_tensor = Variable(
                            torch.from_numpy(
                                image_features).type(torch.FloatTensor))
                    prediction = model(lookup_tensor, image_features_tensor)
                    target = Variable(torch.zeros(1))
                    if image_id == sample['target_img_id']:
                        target = Variable(torch.ones(1))
                    loss = F.l1_loss(prediction, target)
                    train_loss += loss.data[0]
                    print(train_loss)
                    # backward pass
                    model.zero_grad()
                    loss.backward()

                    # update weights
                    optimizer.step()




if __name__ == '__main__':

    easy_dataset = EasyDataset(
            data_directory="./data/",
            training_file="IR_train_easy.json",
            image_mapping_file="IR_image_features2id.json",
            image_feature_file="IR_image_features.h5",
            )
    train_network(easy_dataset, num_epochs=1)
