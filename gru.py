from data_loader import SimpleDataset
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np


class GRU(nn.Module):
    def __init__(self, vocab_size, loss_fn=None, hidden_layer_dim=256, embedding_space=150, use_cuda=False, n_layers=1):
        super().__init__()
        self.hidden_layer_dim = hidden_layer_dim
        self.n_layers = n_layers
        self.embedding_space = embedding_space

        self.embeddings = nn.Embedding(vocab_size, embedding_space)
        self.gru = nn.GRU(embedding_space, hidden_layer_dim, n_layers, batch_first=True)

        # self.hidden_layer = nn.Linear(
        #         embedding_space+2048,
        #         hidden_layer_dim)

        self.output_layer = nn.Linear(hidden_layer_dim+2048, 1)

        self.use_cuda = use_cuda
        self.float_type = torch.FloatTensor
        self.long_type = torch.LongTensor

        if use_cuda:
            print("Using cuda")
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor
            self.cuda()

        if loss_fn is None:
            self.loss_fn = torch.nn.MSELoss(size_average=True)
        else:
            self.loss_fn = loss_fn


    def forward(self, sentences, sentences_mask, img_inputs):
        batch_size = sentences.data.shape[0]
        sequence_size = sentences.data.shape[1]
        embeds = self.embeddings(sentences)

        # print(sentences)
        # print(sentences_mask)

        packed_embedding = pack_padded_sequence(embeds.view(batch_size, -1, self.embedding_space), sentences_mask, batch_first=True)
        outputs, h_gru = self.gru(packed_embedding)

        inputs = torch.cat((h_gru[0,:,:], img_inputs), dim=1)

        probability = F.sigmoid(self.output_layer(inputs))
        return probability

    def format_sample_into_tensors(self, sample_batch, sample_batch_length, w2i):

        # Forward and backward pass per image, text is fixed
        img_inputs = np.zeros((10*sample_batch_length, 2048))
        outputs = np.zeros((10*sample_batch_length, 1))

        b_index = 0
        w_index = 0

        #Padding
        sentence_max_length = 0
        sentences_mask = []
        for sample in sample_batch:
            temp_sentence_length = len(sample["processed_word_inputs"])
            sentences_mask.append(temp_sentence_length)
            if temp_sentence_length > sentence_max_length:
                sentence_max_length = temp_sentence_length

        word_inputs = np.zeros((sample_batch_length, sentence_max_length)) #Padding zeros

        for sample in sample_batch:
            for index, x in enumerate(sample["processed_word_inputs"]):
                word_inputs[w_index][index] = w2i[x]
            w_index +=1
            # lookup_tensor = word_inputs[b_index]

            for image_id in sample['img_list']:
                # image_features = sample['img_features'][image_id]
                # image_features_tensor = torch.from_numpy(image_features).type(self.float_type)

                # word_inputs[b_index] = lookup_tensor
                # img_inputs[b_index] = image_features_tensor
                img_inputs[b_index] = sample['img_features'][image_id]

                if image_id == sample['target_img_id']:
                    outputs[b_index] = 1.0
                b_index += 1

        #Sort
        sorted_index = len_value_argsort(sentences_mask)

        word_inputs = word_inputs[sorted_index]
        word_inputs = torch.from_numpy(np.asarray(word_inputs, dtype=np.int64).repeat(10, axis=0))
        word_inputs = Variable(word_inputs.type(self.long_type))

        img_inputs = sort_larger_array_by_index(img_inputs, sorted_index)
        img_inputs = torch.from_numpy(np.asarray(img_inputs))
        img_inputs = Variable(img_inputs.type(self.float_type))

        outputs = sort_larger_array_by_index(outputs, sorted_index)
        outputs = torch.from_numpy(np.asarray(outputs))
        outputs = Variable(outputs.type(self.float_type))

        sentences_mask = [sentences_mask[i] for i in sorted_index]
        sentences_mask = np.array(sentences_mask).repeat(10).tolist()

        return word_inputs, sentences_mask, img_inputs, outputs

    def top_rank_accuracy(self, predictions, actual, top_param=3):
        if self.use_cuda:
            predictions = predictions.cpu()
            actual = actual.cpu()

        total_size = len(predictions) / 10.0
        correct = 0
        # actual = [actual[i] for i in sorted_index]

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


def train_gru_network(dataset,
                      validation_dataset,
                      num_epochs=15,
                      batch_size=32,
                      loss_fn=None,
                      save_model=False,
                      use_cuda=False,
                      embedding_space=150,
                      learning_rate=0.0001,
                      hidden_layer_dim=256):

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda x: x,
            shuffle=True)

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss(size_average=True)

    # Actually make the model
    model = GRU(dataset.vocab_size, loss_fn=loss_fn,
                embedding_space=embedding_space, hidden_layer_dim=hidden_layer_dim,
                use_cuda=use_cuda)

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
            word_inputs, sentences_mask, img_inputs, outputs = model.format_sample_into_tensors(sample_batch, len(sample_batch), dataset.w2i)
            count += batch_size
            prediction = model(word_inputs, sentences_mask, img_inputs)

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
        validation_loss, top_rank_1, top_rank_3, top_rank_5 = validate_gru_model(
                                                                dataset.vocab_size,
                                                                dataset.w2i,
                                                                validation_dataset,
                                                                use_cuda=use_cuda,
                                                                model=model)
        top_rank_1_arr[ITER] = top_rank_1
        top_rank_3_arr[ITER] = top_rank_3
        top_rank_5_arr[ITER] = top_rank_5

    if save_model:
        torch.save(model.state_dict(), "data/gru.pt")

    return model, top_rank_1_arr, top_rank_3_arr, top_rank_5_arr


def validate_gru_model(vocab_size,
                        w2i, validation_dataset,
                        model_filename="gru.pt",
                        model=None, use_cuda=False,
                        embedding_space=150):

    print("Evaluating model on validation set")
    if model is None:
        print("Loading Saved Model: " + model_filename)
        model = GRU(vocab_size, embedding_space=embedding_space, use_cuda=use_cuda)
        if not use_cuda:
            #loading a model compiled with gpu on a machine that does not have a gpu
            model.load_state_dict(torch.load("data/"+model_filename, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load("data/"+model_filename))
            model = model.cuda()

    torch.cuda.empty_cache()
    model = model.cpu()
    model.float_type = torch.FloatTensor
    model.long_type = torch.LongTensor

    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=1, collate_fn=lambda x: x)
    outputs = None
    prediction = None

    for batch in val_dl:
        word_inputs, sentences_mask, img_inputs, outputs_batch = model.format_sample_into_tensors(batch, len(batch), w2i)
        pred = model(word_inputs, sentences_mask, img_inputs)
        if outputs is None or prediction is None:
            outputs = outputs_batch
            prediction = pred
        else:
            outputs = torch.cat((outputs, outputs_batch), dim=0)
            prediction = torch.cat((prediction, pred), dim=0)

    top_rank_1 = model.top_rank_accuracy(prediction, outputs, top_param=1)
    top_rank_3 = model.top_rank_accuracy(prediction, outputs, top_param=3)
    top_rank_5 = model.top_rank_accuracy(prediction, outputs, top_param=5)

    loss = model.loss_fn(prediction, outputs)
    print(f"Validation Loss : {loss.data[0]}")
    if use_cuda:
        model = model.cuda()
        model.float_type = torch.cuda.FloatTensor
        model.long_type = torch.cuda.LongTensor

    return loss.data[0], top_rank_1, top_rank_3, top_rank_5


def sort_larger_array_by_index(arr, sorted_index):
    return np.array([arr[index*10:index*10+10] for index in sorted_index]).reshape(arr.shape)


def len_value_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: seq[x], reverse=True)

if __name__=="__main__":
    use_cuda = torch.cuda.is_available()

    dataset = SimpleDataset(
            training_file="IR_train_easy.json",
            preprocessing=True,
            preprocessed_data_filename="easy_training_processed_with_questions"
            )

    validation_dataset = SimpleDataset(
            training_file="IR_val_easy.json",
            preprocessing=True,
            preprocessed_data_filename="easy_val_processed_with_questions"
    )

    model, top_rank_1_arr, \
    top_rank_3_arr, top_rank_5_arr = train_gru_network(
                                                dataset,
                                                validation_dataset,
                                                num_epochs=30,
                                                batch_size=256,
                                                embedding_space=300,
                                                hidden_layer_dim=256,
                                                learning_rate=0.001,
                                                use_cuda=use_cuda)

    from run_experiment import save_top_ranks
    save_top_ranks(top_rank_1_arr, top_rank_3_arr, top_rank_5_arr, filename="GRU_NORMAL_PREPROCESS_W_QUESTIONS.pt")

    #  graph_top_ranks(top_rank_1_arr, top_rank_3_arr, top_rank_5_arr)
