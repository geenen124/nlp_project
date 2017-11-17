from data_loader import EasyDataset
import csv


if __name__ == '__main__':

    easy_dataset = EasyDataset(
            data_directory="./data/",
            training_file="IR_train_easy.json",
            image_mapping_file="IR_image_features2id.json",
            image_feature_file="IR_image_features.h5",
            )
