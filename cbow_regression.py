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
loss_fn = torch.nn.MSELoss(size_average=True)
# loss_fn = torch.nn.SmoothL1Loss(size_average=True)
# loss_fn = torch.nn.CosineEmbeddingLoss(margin=0,size_average=False)

# Outputs image features from words
class CBOW_REG(nn.Module):
    def __init__(self, vocab_size, embedding_space, image_feature_size, hidden_layer_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_space)
        self.hidden_layer = nn.Linear(
                embedding_space,
                hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, image_feature_size)
        # self.activation = nn.SELU(inplace=True)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        bow = torch.sum(embeddings, 1)

        output_hidden = self.hidden_layer(bow)
        predicted_image_features = self.output_layer(output_hidden)
        return predicted_image_features


def format_sample_into_tensors(sample_batch, sample_batch_length, w2i):
    global float_type, long_type

    # Forward and backward pass per image, text is fixed
    b_index = 0

    #Padding
    sentence_max_length = 0
    for sample in sample_batch:
        if len(sample["processed_word_inputs"]) > sentence_max_length:
            sentence_max_length = len(sample["processed_word_inputs"])

    word_inputs = torch.zeros((sample_batch_length, sentence_max_length)) #Padding zeros
    outputs = torch.zeros((sample_batch_length, 2048))

    for sample in sample_batch:
        for index, x in enumerate(sample["processed_word_inputs"]):
            word_inputs[b_index][index] = w2i[x]

        outputs[b_index] = torch.from_numpy(
            sample["target_img_features"]).type(float_type)

        b_index +=1

    inputs = Variable(word_inputs.type(long_type))

    outputs = Variable(outputs.type(float_type))

    return inputs, outputs


def train_network(dataset, num_epochs=15, batch_size=32, save_model=False):
    global use_cuda, loss_fn

    image_feature_size = 2048
    embedding_space = 150

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda x: x,
            shuffle=True)

    # Actually make the model
    model = CBOW_REG(dataset.vocab_size, embedding_space, image_feature_size, hidden_layer_dim=256)
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
        for sample_batch in dataloader:
            # Forward and backward pass per image, text is fixed
            inputs, outputs = format_sample_into_tensors(sample_batch, batch_size, dataset.w2i)
            count += batch_size
            prediction = model(inputs)

            loss = loss_fn(prediction, outputs)
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
        torch.save(model.state_dict(), "data/cbow_reg.pt")

    graph_top_ranks(top_rank_1_arr, top_rank_3_arr, top_rank_5_arr)

def top_rank_accuracy(predictions, dataset, top_param=3):
    global use_cuda, loss_fn
    if use_cuda:
        predictions = predictions.cpu()
        actual = actual.cpu()

    total_size = len(predictions)
    correct = 0

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

            image_loss_from_prediction = loss_fn(prediction, image_features_tensor)
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

def validate_saved_model(vocab_size, w2i, model_filename="cbow_reg.pt", model=None):
    global use_cuda, loss_fn
    # Loading a model
    valid_dataset = SimpleDataset(
            training_file="IR_val_easy.json",
            preprocessing=True,
            preprocessed_data_filename="easy_val_processed"
    )

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

    inputs, outputs = format_sample_into_tensors(valid_dataset, len(valid_dataset), w2i)

    predictions = model(inputs)

    top_rank_1 = top_rank_accuracy(predictions, valid_dataset, top_param=1)
    top_rank_3 = top_rank_accuracy(predictions, valid_dataset, top_param=3)
    top_rank_5 = top_rank_accuracy(predictions, valid_dataset, top_param=5)

    loss = loss_fn(predictions, outputs)
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
            training_file="IR_train_easy.json",
            preprocessing=True,
            preprocessed_data_filename="easy_training_processed",
            )


    #Train Network
    train_network(
            easy_dataset,
            num_epochs=15,
            batch_size=64)

    #Validate on validation set:
    # validate_saved_model(easy_dataset.vocab_size, easy_dataset.w2i)
