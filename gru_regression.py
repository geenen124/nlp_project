from data_loader import SimpleDataset
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np

from run_experiment import graph_top_ranks

# Outputs image features from words
class GRU_REG(nn.Module):
    def __init__(self, vocab_size, loss_fn=None, hidden_layer_dim=256, embedding_space=150, use_cuda=False, n_layers=1):
        super().__init__()
        self.hidden_layer_dim = hidden_layer_dim
        self.n_layers = n_layers
        self.embedding_space = embedding_space

        self.embeddings = nn.Embedding(vocab_size, embedding_space)
        self.gru = nn.GRU(embedding_space, hidden_layer_dim, n_layers, batch_first=True)

        self.output_layer = nn.Linear(hidden_layer_dim, 2048)

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

        self.iter = 0

    def forward(self, sentences, sentences_mask):

        batch_size = sentences.data.shape[0]
        sequence_size = sentences.data.shape[0]
        embeds = self.embeddings(sentences)

        packed_embedding = pack_padded_sequence(embeds.view(batch_size, -1, self.embedding_space), sentences_mask, batch_first=True)
        outputs, h_gru = self.gru(packed_embedding, self.init_hidden(batch_size))

        ## unpacking:
        # outputs_pad, output_lengths = pad_packed_sequence(outputs, batch_first=True)
        # output_lengths = Variable(torch.LongTensor(output_lengths))
        # last_out = torch.gather(outputs_pad, 1, output_lengths.view(-1, 1, 1).expand(batch_size, 1, self.hidden_layer_dim)-1).view(batch_size, self.hidden_layer_dim)
        # last_out =  h_gru[0,:,:]

        predicted_image_features = self.output_layer(h_gru[0,:,:])
        return predicted_image_features


    def init_hidden(self, batch_size):
        ih = Variable(torch.zeros(batch_size, self.n_layers, self.hidden_layer_dim))
        if self.use_cuda:
            ih = ih.cuda()
        return ih

    def format_sample_into_tensors(self, sample_batch, sample_batch_length, w2i):
        # Forward and backward pass per image, text is fixed
        b_index = 0

        #Padding
        sentence_max_length = 0
        sentences_mask = []
        for sample in sample_batch:
            temp_sentence_length = len(sample["processed_word_inputs"])
            sentences_mask.append(temp_sentence_length)
            if temp_sentence_length > sentence_max_length:
                sentence_max_length = temp_sentence_length

        word_inputs = np.zeros((sample_batch_length, sentence_max_length)) #Padding zeros
        outputs = np.zeros((sample_batch_length, 2048))

        for sample in sample_batch:
            for index, x in enumerate(sample["processed_word_inputs"]):
                word_inputs[b_index][index] = w2i[x]

            outputs[b_index] = sample["target_img_features"] #torch.from_numpy().type(self.float_type)

            b_index +=1

        #Sort
        sorted_index = len_value_argsort(sentences_mask)

        word_inputs = [word_inputs[i] for i in sorted_index]
        word_inputs = torch.from_numpy(np.array(word_inputs, dtype=np.int64))
        inputs = Variable(word_inputs.type(self.long_type))

        outputs = [outputs[i] for i in sorted_index]
        outputs = torch.from_numpy(np.array(outputs, dtype=np.float64))
        outputs = Variable(outputs.type(self.float_type))

        sentences_mask = [sentences_mask[i] for i in sorted_index]

        return inputs, sentences_mask, outputs

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


def train_gru_reg_network(dataset,
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
    model = GRU_REG(dataset.vocab_size, loss_fn=loss_fn,
                    embedding_space=embedding_space,
                    hidden_layer_dim=hidden_layer_dim, use_cuda=use_cuda)


    # hx = Variable(torch.randn(1, batch_size, hidden_layer_dim))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = 0.0

    top_rank_1_arr = np.zeros(num_epochs)
    top_rank_3_arr = np.zeros(num_epochs)
    top_rank_5_arr = np.zeros(num_epochs)

    for ITER in range(num_epochs):
        print(f"Training Loss for {ITER} :  {train_loss}")
        train_loss = 0.0
        count = 0
        # model.init_hidden(batch_size)

        for sample_batch in dataloader:
            # Forward and backward pass per image, text is fixed
            inputs, sentences_mask, outputs = model.format_sample_into_tensors(sample_batch, batch_size, dataset.w2i)
            count += batch_size
            prediction = model(inputs, sentences_mask)

            loss = model.loss_fn(prediction, outputs)
            if use_cuda:
                loss = loss.cuda()
            train_loss += loss.data[0]

            print(f"Loss : {loss.data[0]} \t Count: {count}", end="\r")

            # backward pass
            model.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()

            # update weights
            optimizer.step()

        print("\n")
        validation_loss, top_rank_1, top_rank_3, top_rank_5 = validate_gru_reg_model(
                                                                dataset.vocab_size,
                                                                dataset.w2i,
                                                                validation_dataset,
                                                                model=model)
        top_rank_1_arr[ITER] = top_rank_1
        top_rank_3_arr[ITER] = top_rank_3
        top_rank_5_arr[ITER] = top_rank_5

    if save_model:
        torch.save(model.state_dict(), "data/gru_reg.pt")

    return model, top_rank_1_arr, top_rank_3_arr, top_rank_5_arr


def validate_gru_reg_model(vocab_size, w2i, validation_dataset, model_filename="gru_reg.pt",
                        model=None, embedding_space = 150):

    print("Evaluating model on validation set")
    if model is None:
        print("Loading Saved Model: " + model_filename)
        model = GRU_REG(vocab_size, 2048, hidden_layer_dim=256)
        if not use_cuda:
            #loading a model compiled with gpu on a machine that does not have a gpu
            model.load_state_dict(torch.load("data/"+model_filename, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load("data/"+model_filename))
            model = model.cuda()

    inputs, sentences_mask, outputs = model.format_sample_into_tensors(validation_dataset, len(validation_dataset), w2i)
    # model.init_hidden(len(validation_dataset))
    predictions = model(inputs, sentences_mask)

    top_rank_1 = model.top_rank_accuracy(predictions, validation_dataset, top_param=1)
    top_rank_3 = model.top_rank_accuracy(predictions, validation_dataset, top_param=3)
    top_rank_5 = model.top_rank_accuracy(predictions, validation_dataset, top_param=5)

    loss = model.loss_fn(predictions, outputs)
    print(f"Validation Loss : {loss.data[0]}")

    return loss.data[0], top_rank_1, top_rank_3, top_rank_5

def len_value_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: seq[x], reverse=True)


use_cuda = torch.cuda.is_available()

dataset = SimpleDataset(
        training_file="IR_train_easy.json",
        preprocessing=False,
        preprocessed_data_filename="easy_training_unprocessed"
        )

validation_dataset = SimpleDataset(
        training_file="IR_val_easy.json",
        preprocessing=False,
        preprocessed_data_filename="easy_val_unprocessed"
)

model, top_rank_1_arr, \
top_rank_3_arr, top_rank_5_arr = train_gru_reg_network(
                                            dataset,
                                            validation_dataset,
                                            num_epochs=5,
                                            batch_size=256,
                                            embedding_space=300,
                                            hidden_layer_dim=256,
                                            learning_rate=0.001,
                                            use_cuda=use_cuda)

graph_top_ranks(top_rank_1_arr, top_rank_3_arr, top_rank_5_arr)
