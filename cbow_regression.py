from data_loader import SimpleDataset
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

use_cuda = False
float_type = torch.FloatTensor
long_type = torch.LongTensor

# Outputs image features from words
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_space, image_feature_size, hidden_layer_dim):
        super().__init__()
        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_space, mode='mean')
        self.hidden_layer = nn.Linear(
                embedding_space,
                hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, image_feature_size)

    def encode_words(self, word_inputs):
        offsets = Variable(torch.LongTensor([0]))
        bow = self.embeddings(word_inputs, offsets)
        return torch.squeeze(bow)

    def forward(self, inputs):
        output_hidden = self.hidden_layer(inputs)
        predicted_image_features = F.sigmoid(self.output_layer(output_hidden))
        return predicted_image_features

def format_sample_into_tensors(sample_batch,
                            sample_batch_length,
                            embedding_space, w2i,
                            model, dataset):
    global float_type, long_type

    # Forward and backward pass per image, text is fixed
    inputs = torch.zeros((sample_batch_length, embedding_space))
    outputs = torch.zeros((sample_batch_length, 2048))

    b_index = 0
    for sample in sample_batch:
        lookup_tensor = Variable(
            torch.LongTensor([w2i[x] for x in sample["processed_word_inputs"]]).type(long_type))
        bow = model.encode_words(lookup_tensor)
        inputs[b_index] = bow.data
        outputs[b_index] = torch.from_numpy(
            sample["target_img_features"]).type(float_type)

    inputs = Variable(inputs.type(float_type))
    outputs = Variable(outputs.type(float_type))
    return inputs, outputs


def train_network(dataset, num_epochs=1000, batch_size=32, save_model=False):
    global use_cuda

    image_feature_size = 2048
    embedding_space = 150
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda x: x,
            shuffle=True)

    # Actually make the model
    model = CBOW(dataset.vocab_size, embedding_space, image_feature_size, hidden_layer_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loss = 0.0

    if use_cuda:
        model = model.cuda()

    top_rank_1_arr = np.zeros(num_epochs)
    top_rank_3_arr = np.zeros(num_epochs)
    top_rank_5_arr = np.zeros(num_epochs)

    for ITER in range(num_epochs):
        print(f"Training Loss for {ITER} :  {train_loss}")
        train_loss = 0.0
        count = 0
        # for sample_batch in dataloader:
        for sample_batch in dataset:

            # Forward and backward pass per image, text is fixed
            inputs, outputs = format_sample_into_tensors(sample_batch, batch_size, embedding_space, dataset.w2i, model, dataset)
            count += batch_size
            prediction = model(inputs)

            print("First prediction")
            print(prediction.data[0])

            print("Actual")
            print(outputs.data[0])

            loss = F.smooth_l1_loss(prediction, outputs)
            if use_cuda:
                loss = loss.cuda()
            train_loss += loss.data[0]

            print(f"Loss : {loss.data[0]} \t Count: {count}", end="\r")

            # backward pass
            model.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

        print("\n")
        validation_loss, top_rank_1, top_rank_3, top_rank_5 = validate_saved_model(
                                                                dataset.vocab_size,
                                                                dataset.w2i,
                                                                model=model)
        top_rank_1_arr[ITER] = top_rank_1
        top_rank_3_arr[ITER] = top_rank_3
        top_rank_5_arr[ITER] = top_rank_5

    if save_model:
        torch.save(model.state_dict(), "data/cbow.pt")

    graph_top_ranks(top_rank_1_arr, top_rank_3_arr, top_rank_5_arr)

def top_rank_accuracy(predictions, dataset, top_param=3):
    global use_cuda
    if use_cuda:
        predictions = predictions.cpu()
        actual = actual.cpu()

    total_size = len(predictions)
    correct = 0
    predictions_np = predictions.data.numpy()

    for index, prediction in  enumerate(predictions):
        sample = dataset[index]
        actual_slice = np.zeros(10)
        prediction_slice = np.zeros(10) #loss from each image
        b_index = 0

        for image_id in sample['img_list']:
            image_features = sample['img_features'][image_id]
            image_features_tensor = Variable(
                    torch.from_numpy(
                        image_features).type(float_type))
            image_loss_from_prediction = F.smooth_l1_loss(prediction, image_features_tensor)
            prediction_slice[b_index] = 1.0 - image_loss_from_prediction.data[0]

            if image_id == sample['target_img_id']:
                actual_slice[b_index] = 1.0
            b_index += 1

        #do argmax on n (top_param) indexes
        prediction_indexes = prediction_slice.flatten().argsort()[-top_param:][::-1]
        if actual_slice[prediction_indexes].any():
            correct += 1

    print(f"{correct} correct out of {total_size}")
    return float(correct) / total_size

def validate_saved_model(vocab_size, w2i, model_filename="cbow.pt", model=None):
    global use_cuda
    # Loading a model
    embedding_space = 150
    print("Evaluating model on validation set")
    if model is None:
        print("Loading Saved Model: " + model_filename)
        model = CBOW(vocab_size, embedding_space, 2048, hidden_layer_dim=256)
        if not use_cuda:
            #loading a model compiled with gpu on a machine that does not have a gpu
            model.load_state_dict(torch.load("data/"+model_filename, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load("data/"+model_filename))
            model = model.cuda()

    valid_dataset = SimpleDataset(
            data_directory="./data/",
            training_file="IR_val_easy.json",
            image_mapping_file="IR_image_features2id.json",
            image_feature_file="IR_image_features.h5",
            preprocessing=True,
            preprocessed_data_filename="easy_val_processed_with_questions"
    )

    inputs, outputs = format_sample_into_tensors(valid_dataset, len(valid_dataset), embedding_space, w2i, model, valid_dataset)

    predictions = model(inputs)

    top_rank_1 = top_rank_accuracy(predictions, valid_dataset, top_param=1)
    top_rank_3 = top_rank_accuracy(predictions, valid_dataset, top_param=3)
    top_rank_5 = top_rank_accuracy(predictions, valid_dataset, top_param=5)

    loss = F.smooth_l1_loss(predictions, outputs)
    print(f"Validation Loss : {loss.data[0]}")

    return loss.data[0], top_rank_1, top_rank_3, top_rank_5


def graph_top_ranks(top_rank_1_arr, top_rank_3_arr, top_rank_5_arr):
    x_axis = np.arange(len(top_rank_1_arr))+1
    plt.plot(x_axis, top_rank_1_arr, label="top-1")
    plt.plot(x_axis, top_rank_3_arr, label="top-3")
    plt.plot(x_axis, top_rank_5_arr, label="top-5")
    # plt.axis([1, len(top_rank_1_arr), 0, 1])
    plt.legend()#bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.title('Top Rank Probability')
    plt.show()


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("Using cuda")
        float_type = torch.cuda.FloatTensor
        long_type = torch.cuda.LongTensor

    easy_dataset = SimpleDataset(
            data_directory="./data/",
            training_file="IR_train_easy.json",
            image_mapping_file="IR_image_features2id.json",
            image_feature_file="IR_image_features.h5",
            preprocessing=True,
            preprocessed_data_filename="easy_training_processed_with_questions"
            )

    #Train Network
    train_network(
            easy_dataset,
            num_epochs=1000,
            batch_size=1)

    #Validate on validation set:
    # validate_saved_model(easy_dataset.vocab_size, easy_dataset.w2i)
