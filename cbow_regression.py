from data_loader import SimpleDataset
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Outputs image features from words
class CBOW_REG(nn.Module):
    def __init__(self, vocab_size, loss_fn=None, hidden_layer_dim=256, embedding_space=150, use_cuda=False):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_space)
        self.hidden_layer = nn.Linear(
                embedding_space,
                hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, 2048)
        # self.activation = nn.SELU(inplace=True)

        self.use_cuda = use_cuda
        self.float_type = torch.FloatTensor
        self.long_type = torch.LongTensor

        if use_cuda:
            print("Using cuda")
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor
            self.cuda()

        if loss_fn is None:
            self.loss_fn = torch.nn.SmoothL1Loss(size_average=True)
        else:
            self.loss_fn = loss_fn

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        bow = torch.sum(embeddings, 1)

        output_hidden = self.hidden_layer(bow)
        predicted_image_features = self.output_layer(output_hidden)
        return predicted_image_features


    def format_sample_into_tensors(self, sample_batch, sample_batch_length, w2i):
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
                sample["target_img_features"]).type(self.float_type)

            b_index +=1

        inputs = Variable(word_inputs.type(self.long_type))

        outputs = Variable(outputs.type(self.float_type))

        return inputs, outputs

    def top_rank_accuracy(self, predictions, dataset, top_param=3):
        if self.use_cuda:
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
                            image_features).type(self.float_type))

                image_loss_from_prediction = self.loss_fn(prediction, image_features_tensor)
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

def train_cbow_reg_network(dataset,
                          validation_dataset,
                          loss_fn=None,
                          embedding_space=150,
                          num_epochs=15,
                          batch_size=32,
                          save_model=False,
                          learning_rate = 0.0001,
                          hidden_layer_dim=256,
                          use_cuda=False):

    if loss_fn is None:
        loss_fn = torch.nn.SmoothL1Loss(size_average=True)


    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda x: x,
            shuffle=True)

    # Actually make the model
    model = CBOW_REG(dataset.vocab_size, loss_fn=loss_fn,
                    embedding_space=embedding_space,
                    hidden_layer_dim=hidden_layer_dim, use_cuda=use_cuda)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = 0.0

    top_rank_1_arr = np.zeros(num_epochs)
    top_rank_3_arr = np.zeros(num_epochs)
    top_rank_5_arr = np.zeros(num_epochs)

    for ITER in range(num_epochs):
        print(f"Training Loss for {ITER} :  {train_loss}")
        train_loss = 0.0
        count = 0
        for sample_batch in dataloader:
            # Forward and backward pass per image, text is fixed
            inputs, outputs = model.format_sample_into_tensors(sample_batch, len(sample_batch), dataset.w2i)
            count += batch_size
            prediction = model(inputs)

            loss = model.loss_fn(prediction, outputs)
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
        validation_loss, top_rank_1, top_rank_3, top_rank_5 = validate_cbow_reg_model(
                                                                dataset.vocab_size,
                                                                dataset.w2i,
                                                                validation_dataset,
                                                                model=model)
        top_rank_1_arr[ITER] = top_rank_1
        top_rank_3_arr[ITER] = top_rank_3
        top_rank_5_arr[ITER] = top_rank_5

    if save_model:
        torch.save(model.state_dict(), "data/cbow_reg.pt")

    return model, top_rank_1_arr, top_rank_3_arr, top_rank_5_arr


def validate_cbow_reg_model(vocab_size, w2i, validation_dataset, model_filename="cbow_reg.pt",
                        model=None, embedding_space = 150):

    print("Evaluating model on validation set")
    if model is None:
        print("Loading Saved Model: " + model_filename)
        model = CBOW_REG(vocab_size, 2048, hidden_layer_dim=256)
        if not use_cuda:
            #loading a model compiled with gpu on a machine that does not have a gpu
            model.load_state_dict(torch.load("data/"+model_filename, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load("data/"+model_filename))
            model = model.cuda()

    inputs, outputs = model.format_sample_into_tensors(validation_dataset, len(validation_dataset), w2i)

    predictions = model(inputs)

    top_rank_1 = model.top_rank_accuracy(predictions, validation_dataset, top_param=1)
    top_rank_3 = model.top_rank_accuracy(predictions, validation_dataset, top_param=3)
    top_rank_5 = model.top_rank_accuracy(predictions, validation_dataset, top_param=5)

    loss = model.loss_fn(predictions, outputs)
    print(f"Validation Loss : {loss.data[0]}")

    return loss.data[0], top_rank_1, top_rank_3, top_rank_5
