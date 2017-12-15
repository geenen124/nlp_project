import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import SimpleDataset
# from cbow import CBOW as MODEL, train_cbow_network as train_network, validate_cbow_model as validate_model
# from gru import GRU as MODEL, train_gru_network2 as train_network, validate_gru_model2 as validate_model
from gru_regression import GRU_REG as MODEL, train_gru_reg_network as train_network, validate_gru_reg_model as validate_model
# from cbow_regression import CBOW_REG as MODEL, train_cbow_reg_network as train_network, validate_cbow_reg_model as validate_model
import pickle
modelname= "GRU_REG"
from IPython import embed
import gc

def save_top_ranks(top_rank_1_arr, top_rank_3_arr, top_rank_5_arr, filename):
    top_dict = {
            "top_1": top_rank_1_arr,
            "top_3": top_rank_3_arr,
            "top_5": top_rank_5_arr,
            }
    pickled_file = open(filename, 'wb')
    pickle.dump(top_dict, pickled_file)

def save_performance_log(performance_log, filename):
    pickled_file = open(filename, 'wb')
    pickle.dump(performance_log, pickled_file)

def graph_top_ranks(top_rank_1_arr, top_rank_3_arr, top_rank_5_arr):
    x_axis = np.arange(len(top_rank_1_arr))+1
    plt.plot(x_axis, top_rank_1_arr, label="top-1")
    plt.plot(x_axis, top_rank_3_arr, label="top-3")
    plt.plot(x_axis, top_rank_5_arr, label="top-5")
    plt.legend()

    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.title('Top Rank Probability')
    plt.show()
    # graph_top_ranks(top_rank_1_arr, top_rank_3_arr, top_rank_5_arr)

def save_model(model_type, hidden_layer_dim, embedding_space, learning_rate, loss_fn_name, model):
    # if model.embedding_space != embedding_space:
    #     print("model embed != embed space")
    #     embed()
    filename = model_type+"_"+str(hidden_layer_dim)+"_"+str(embedding_space)+"_"+str(learning_rate)+"_"+loss_fn_name
    torch.save(model.state_dict(), "models/"+filename+".pt")

def get_model(vocab_size, model_type, hidden_layer_dim, embedding_space, learning_rate, loss_fn_name):
    filename = model_type+"_"+str(hidden_layer_dim)+"_"+str(embedding_space)+"_"+str(learning_rate)+"_"+loss_fn_name
    if os.path.isfile("models/"+filename+".pt"):
        if model_type==f"{modelname}_Easy":
            print(f"Loading Model from {filename}")
            model = MODEL(vocab_size, hidden_layer_dim=hidden_layer_dim, embedding_space=embedding_space)
            model.load_state_dict(torch.load("models/"+filename+".pt"))
        return model
    else:
        return None

def evaluate_at_params(dataset, validation_dataset, model_type, hidden_layer_dim, embedding_space, learning_rate, use_cuda, loss_fn_name="SmoothL1Loss"):
    model = get_model(dataset.vocab_size, model_type, hidden_layer_dim, embedding_space, learning_rate, loss_fn_name)
    if model is None:
        model, top_rank_1_arr, \
        top_rank_3_arr, top_rank_5_arr = train_network(
                                                    dataset,
                                                    validation_dataset,
                                                    num_epochs=5,
                                                    batch_size=64,
                                                    embedding_space=embedding_space,
                                                    hidden_layer_dim=hidden_layer_dim,
                                                    learning_rate=learning_rate,
                                                    use_cuda=use_cuda)

        save_model(f"{modelname}_Easy", hidden_layer_dim, embedding_space, learning_rate, loss_fn_name, model)
        return top_rank_1_arr[-1], top_rank_3_arr[-1], top_rank_5_arr[-1]

    else:
        loss, top_rank_1, \
        top_rank_3, top_rank_5 = validate_model(dataset.vocab_size, dataset.w2i, validation_dataset, model=model)
        return top_rank_1, top_rank_3, top_rank_5


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    easy_dataset = SimpleDataset(
            training_file="IR_train_easy.json",
            preprocessing=True,
            preprocessed_data_filename="easy_training_processed_with_questions"
            )

    test_dataset = SimpleDataset(
            training_file="IR_test_easy.json",
            preprocessing=True,
            preprocessed_data_filename="easy_val_processed_with_questions"
    )

    loss_fn = torch.nn.MSELoss(size_average=True)
    loss_fn = torch.nn.SmoothL1Loss(size_average=True)
    Embedding_Spaces = [100, 150, 200, 250, 300, 350]
    Hidden_Dims = [56, 256, 512, 1024]
    Learning_Rates = [0.001, 0.0001, 0.00001]

    best_top_1 = 0
    best_top_params = []
    performance_log = {}
    for e in Embedding_Spaces:
        for h in Hidden_Dims:
            for l in Learning_Rates:
                top_rank_1, top_rank_3, top_rank_5 = evaluate_at_params(easy_dataset, valid_dataset, f"{modelname}_Easy", h, e, l, use_cuda)
                key = f"embd:{e}, h:{h}, l:{l}"
                print("CURRENT")
                print(key)
                performance_log[key] = [top_rank_1, top_rank_3, top_rank_5]

                if top_rank_1 > best_top_1:
                    best_top_1 = top_rank_1
                    best_top_params = [e, h, l]
                    print("BEST TOP PARAMS: ")
                    print(best_top_params)
    print("BEST TOP PARAMS: ")
    print(best_top_params)
    save_performance_log(performance_log, f"./optimization_log_{modelname}_easy.p")


    embedding_space, hidden_layer_dim, learning_rate = best_top_params 
    model, top_rank_1, \
            top_rank_3, top_rank_5 = train_network( easy_dataset,
                                                   valid_dataset,
                                                   num_epochs=30,
                                                   batch_size=128,
                                                   embedding_space=embedding_space,
                                                   hidden_layer_dim=hidden_layer_dim,
                                                   learning_rate=learning_rate,
                                                   use_cuda=use_cuda)
    save_model("GRU_REG_EASY",
            hidden_layer_dim=hidden_layer_dim,
            embedding_space=embedding_space,
            learning_rate=learning_rate,
            loss_fn_name="mse",
            model=model)

    save_top_ranks(top_rank_1, top_rank_3, top_rank_5, f"./results_{modelname}_easy_best_params_{best_top_params}.p")
    print(top_rank_1)
    print(top_rank_3)
    print(top_rank_5)

    hard_dataset = SimpleDataset(
            training_file="IR_train_hard.json",
            preprocessing=True,
            preprocessed_data_filename="hard_training_processed_with_questions"
            )

    valid_hard_dataset = SimpleDataset(
            training_file="IR_val_hard.json",
            preprocessing=True,
            preprocessed_data_filename="hard_val_processed_with_questions"
    )


    model, top_rank_1, \
            top_rank_3, top_rank_5 = train_network( hard_dataset,
                                                   valid_hard_dataset,
                                                   num_epochs=30,
                                                   batch_size=128,
                                                   embedding_space=embedding_space,
                                                   hidden_layer_dim=hidden_layer_dim,
                                                   learning_rate=learning_rate,
                                                   use_cuda=use_cuda)
    save_model("GRU_REG_HARD",
            hidden_layer_dim=hidden_layer_dim,
            embedding_space=embedding_space,
            learning_rate=learning_rate,
            loss_fn_name="mse",
            model=model)
    save_top_ranks(top_rank_1, top_rank_3, top_rank_5, f"./results_{modelname}_hard_best_params_{best_top_params}.p")
    print(top_rank_1)
    print(top_rank_3)
    print(top_rank_5)
