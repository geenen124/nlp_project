from cbow import CBOW, train_cbow_network, validate_cbow_model
from gru import GRU, train_gru_network2, validate_gru_model2
from gru_regression import GRU_REG, train_gru_reg_network, validate_gru_reg_model
from cbow_regression import CBOW_REG, train_cbow_reg_network, validate_cbow_reg_model
from data_loader import SimpleDataset

def test_easy_and_hard(model_easy, model_hard, model_easy_file, model_hard_file, validation_func, easy_dataset, easy_testset, hard_dataset, hard_testset):
    model.load_state_dict(torch.load(f"models/{model_easy_file}"))

    loss, top_rank_1, \
    top_rank_3, top_rank_5 = validation_func(easy_dataset.vocab_size, easy_dataset.w2i, easy_testset, model=model)

    print("EASY RESULTS")
    print(top_rank_1)
    print(top_rank_3)
    print(top_rank_5)

    model.load_state_dict(torch.load(f"models/{model_hard_file}"))

    loss, top_rank_1, \
    top_rank_3, top_rank_5 = validation_func(hard_dataset.vocab_size, hard_dataset.w2i, hard_testset, model=model)

    print("HARD RESULTS")
    print(top_rank_1)
    print(top_rank_3)
    print(top_rank_5)


if __name__ == '__main__':
    easy_dataset = SimpleDataset(
            training_file="IR_train_easy.json",
            preprocessing=True,
            preprocessed_data_filename="easy_training_processed_with_questions"
            )

    easy_dataset_cbow = SimpleDataset(
            training_file="IR_train_easy.json",
            preprocessing=True,
            preprocessed_data_filename="easy_training_unprocessed" # Saved to wrong file, didn't want to wait an hour to reprocess
            )

    easy_test_dataset = SimpleDataset(
            training_file="IR_test_easy.json",
            preprocessing=True,
            preprocessed_data_filename="easy_test_processed_with_questions"
    )

    hard_dataset = SimpleDataset(
            training_file="IR_train_hard.json",
            preprocessing=True,
            preprocessed_data_filename="hard_training_processed_with_questions"
            )

    # Same issue with CBOW as above (filenames)
    hard_dataset_cbow = SimpleDataset(
            training_file="IR_train_hard.json",
            preprocessing=True,
            preprocessed_data_filename="hard_training_unprocessed_with_questions"
            )

    hard_test_dataset = SimpleDataset(
            training_file="IR_test_hard.json",
            preprocessing=True,
            preprocessed_data_filename="hard_test_processed_with_questions"
    )

    # GRU REGRESSION
    best_top_params = [350, 1024, 0.001]
    embedding_space, hidden_layer_dim, learning_rate = best_top_params
    model_easy_file ="GRU_REG_EASY_1024_350_0.001_mse.pt"
    model_easy = GRU_REG(easy_dataset.vocab_size, hidden_layer_dim=hidden_layer_dim, embedding_space=embedding_space)
    model_hard_file ="GRU_REG_HARD_1024_350_0.001_mse.pt"
    model_hard = GRU_REG(hard_dataset.vocab_size, hidden_layer_dim=hidden_layer_dim, embedding_space=embedding_space)
    test_easy_and_hard(model_easy, model_hard, model_easy_file, model_hard_file, validate_gru_reg_model, easy_dataset, easy_testset, hard_dataset, hard_testset)

    # GRU
    best_top_params = [300, 56, 0.0001]
    embedding_space, hidden_layer_dim, learning_rate = best_top_params
    model_easy_file = "GRU_EASY_56_300_0.0001_mse.pt"
    model_easy = GRU(easy_dataset.vocab_size, hidden_layer_dim=hidden_layer_dim, embedding_space=embedding_space)
    model_hard_file = "GRU_HARD_56_300_0.0001_mse.pt"
    model_hard = GRU(hard_dataset.vocab_size, hidden_layer_dim=hidden_layer_dim, embedding_space=embedding_space)
    test_easy_and_hard(model_easy, model_hard, model_easy_file, model_hard_file, validate_gru_model, easy_dataset, easy_testset, hard_dataset, hard_testset)

    # CBOW NAIVE
    best_top_params = [150, 256, 0.0001]
    embedding_space, hidden_layer_dim, learning_rate = best_top_params
    model_easy_file = "CBOW_NAIVE_EASY_256_150_0.0001_smooth_l1.pt"
    model_easy = CBOW(easy_dataset_cbow.vocab_size, hidden_layer_dim=hidden_layer_dim, embedding_space=embedding_space)
    model_hard_file = "CBOW_NAIVE_HARD_256_150_0.0001_smooth_l1.pt"
    model_hard = CBOW(hard_dataset_cbow.vocab_size, hidden_layer_dim=hidden_layer_dim, embedding_space=embedding_space)
    test_easy_and_hard(model_easy, model_hard, model_easy_file, model_hard_file, validate_cbow_model, easy_dataset_cbow, easy_testset, hard_dataset_cbow, hard_testset)

    # CBOW REGRESSION
    best_top_params = [150, 256, 0.0001]
    embedding_space, hidden_layer_dim, learning_rate = best_top_params
    model_easy_file = "CBOW_REG_EASY_256_300_0.001_smooth_l1.pt"
    model_easy = CBOW_REG(easy_dataset_cbow.vocab_size, hidden_layer_dim=hidden_layer_dim, embedding_space=embedding_space)
    model_hard_file = "CBOW_REG_HARD_256_300_0.001_smooth_l1.pt"
    model_hard = CBOW_REG(hard_dataset_cbow.vocab_size, hidden_layer_dim=hidden_layer_dim, embedding_space=embedding_space)
    test_easy_and_hard(model_easy, model_hard, model_easy_file, model_hard_file, validate_cbow_model, easy_dataset_cbow, easy_testset, hard_dataset_cbow, hard_testset)
