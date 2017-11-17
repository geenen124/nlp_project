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

    def encode_words(self, word_inputs):
        embeddings = self.embeddings(word_inputs)
        bow = torch.sum(embeddings, 0)
        return bow

    def forward(self, inputs):
        output_hidden = self.hidden_layer(inputs)
        probability = F.sigmoid(self.output_layer(output_hidden))
        return probability


def train_network(dataset, num_epochs=1000, batch_size=32):
    word_to_index, vocab_size = dataset_to_index_dict_and_size(dataset)
    image_feature_size = 2048
    embedding_space = 150
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda x: x,
            shuffle=True)

    # Actually make the model
    model = CBOW(vocab_size, embedding_space, image_feature_size, hidden_layer_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loss = 0.0

    for ITER in range(num_epochs):
        print(f"Training Loss for {ITER} :  {train_loss}")
        print(len(dataloader)*batch_size)
        train_loss = 0.0
        count = 0
        for sample_batch in dataloader:

            # Forward and backward pass per image, text is fixed
            inputs = torch.zeros((10*batch_size, image_feature_size+embedding_space))
            outputs = torch.zeros((10*batch_size, 1))
            
            b_index = 0
            for sample in sample_batch:
                caption = sample["caption"].lower().split(" ")
                lookup_tensor = Variable(
                    torch.LongTensor([word_to_index[x] for x in caption]))
                bow = model.encode_words(lookup_tensor)

                for image_id in sample['img_list']:
                    image_features = sample['img_features'][image_id]
                    image_features_tensor = Variable(
                            torch.from_numpy(
                                image_features).type(torch.FloatTensor))
                    inputs[b_index] = torch.cat((bow, image_features_tensor)).data

                    if image_id == sample['target_img_id']:
                        outputs[b_index] = 1.0
                    b_index += 1
                count +=1

            inputs = Variable(inputs)
            outputs = Variable(outputs)

            prediction = model(inputs)

            loss = F.l1_loss(prediction, outputs)
            train_loss += loss.data[0]
            print(f"Loss : {loss.data[0]} \t Count: {count}", end="\r")

            # backward pass
            model.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()


        torch.save(model.state_dict(), "data/cbow.pt")

    torch.save(model.state_dict(), "data/cbow.pt")



if __name__ == '__main__':

    easy_dataset = EasyDataset(
            data_directory="./data/",
            training_file="IR_train_easy.json",
            image_mapping_file="IR_image_features2id.json",
            image_feature_file="IR_image_features.h5",
            )
    train_network(easy_dataset, num_epochs=5)


#Loading a model
# model = CBOW(vocab_size, 5, image_feature_size, hidden_layer_dim=256)
# model.load_state_dict(torch.load("data/cbow.pt"))
