from data_loader import SimpleDataset
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class CBOW(nn.Module):
    def __init__(self, vocab_size, loss_fn=None, hidden_layer_dim=256, embedding_space=150, use_cuda=False):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_space)

        self.hidden_layer = nn.Linear(
                embedding_space+2048,
                hidden_layer_dim)

        self.output_layer = nn.Linear(hidden_layer_dim, 1)

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


    def forward(self, word_inputs, img_inputs):
        embeddings = self.embeddings(word_inputs)
        bow = torch.sum(embeddings, 1)

        inputs = torch.cat((bow, img_inputs), dim=1)

        output_hidden = self.hidden_layer(inputs)
        probability = F.sigmoid(self.output_layer(output_hidden))
        return probability

    def format_sample_into_tensors(self, sample_batch, sample_batch_length, w2i):

        # Forward and backward pass per image, text is fixed
        img_inputs = torch.zeros((10*sample_batch_length, 2048))
        outputs = torch.zeros((10*sample_batch_length, 1))

        b_index = 0

        #Padding
        sentence_max_length = 0
        for sample in sample_batch:
            if len(sample["processed_word_inputs"]) > sentence_max_length:
                sentence_max_length = len(sample["processed_word_inputs"])

        word_inputs = torch.zeros((10*sample_batch_length, sentence_max_length)) #Padding zeros

        for sample in sample_batch:
            for index, x in enumerate(sample["processed_word_inputs"]):
                word_inputs[b_index][index] = w2i[x]

            lookup_tensor = word_inputs[b_index]

            for image_id in sample['img_list']:
                image_features = sample['img_features'][image_id]
                image_features_tensor = torch.from_numpy(image_features).type(self.float_type)

                word_inputs[b_index] = lookup_tensor
                img_inputs[b_index] = image_features_tensor

                if image_id == sample['target_img_id']:
                    outputs[b_index] = 1.0
                b_index += 1

        word_inputs = Variable(word_inputs.type(self.long_type))
        img_inputs = Variable(img_inputs.type(self.float_type))
        outputs = Variable(outputs.type(self.float_type))

        return word_inputs, img_inputs, outputs

    def top_rank_accuracy(self, predictions, actual, top_param=3):
        if self.use_cuda:
            predictions = predictions.cpu()
            actual = actual.cpu()

        total_size = len(predictions) / 10.0
        correct = 0
        predictions_np = predictions.data.numpy()
        actual_np = actual.data.numpy()

        for offset in range(int(total_size)):
            start = 10*offset
            end = start + 10
            prediction_slice = predictions_np[start:end]
            actual_slice = actual_np[start:end]

            #do argmax on n (top_param) indexes
            prediction_indexes = prediction_slice.flatten().argsort()[-top_param:][::-1]
            if actual_slice[prediction_indexes].any():
                correct += 1

        print(f"{correct} correct out of {total_size}")
        return float(correct) / total_size


def train_cbow_network(dataset, validation_dataset,
                       num_epochs=1000, batch_size=32,
                       loss_fn=None,
                       save_model=False, use_cuda=False,
                       embedding_space=150, learning_rate=0.0001,
                       hidden_layer_dim=256):

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda x: x,
            shuffle=True)

    if loss_fn is None:
        loss_fn = torch.nn.SmoothL1Loss(size_average=True)

    # Actually make the model
    model = CBOW(dataset.vocab_size, loss_fn=loss_fn, embedding_space=150, hidden_layer_dim=hidden_layer_dim, use_cuda=use_cuda)
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
            word_inputs, img_inputs, outputs = model.format_sample_into_tensors(sample_batch, batch_size, dataset.w2i)
            count += batch_size
            prediction = model(word_inputs, img_inputs)

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
        validation_loss, top_rank_1, top_rank_3, top_rank_5 = validate_cbow_model(
                                                                dataset.vocab_size,
                                                                dataset.w2i,
                                                                validation_dataset,
                                                                model=model)
        top_rank_1_arr[ITER] = top_rank_1
        top_rank_3_arr[ITER] = top_rank_3
        top_rank_5_arr[ITER] = top_rank_5

    if save_model:
        torch.save(model.state_dict(), "data/cbow.pt")

    return model, top_rank_1_arr, top_rank_3_arr, top_rank_5_arr


def validate_cbow_model(vocab_size,
                        w2i, validation_dataset,
                        model_filename="cbow.pt",
                        model=None, use_cuda=False,
                        embedding_space=150):

    print("Evaluating model on validation set")
    if model is None:
        print("Loading Saved Model: " + model_filename)
        model = CBOW(vocab_size, embedding_space=embedding_space, use_cuda=use_cuda)
        if not use_cuda:
            #loading a model compiled with gpu on a machine that does not have a gpu
            model.load_state_dict(torch.load("data/"+model_filename, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load("data/"+model_filename))
            model = model.cuda()

    word_inputs, img_inputs, outputs = model.format_sample_into_tensors(validation_dataset, len(validation_dataset), w2i)

    prediction = model(word_inputs, img_inputs)

    top_rank_1 = model.top_rank_accuracy(prediction, outputs, top_param=1)
    top_rank_3 = model.top_rank_accuracy(prediction, outputs, top_param=3)
    top_rank_5 = model.top_rank_accuracy(prediction, outputs, top_param=5)

    loss = model.loss_fn(prediction, outputs)
    print(f"Validation Loss : {loss.data[0]}")

    return loss.data[0], top_rank_1, top_rank_3, top_rank_5
