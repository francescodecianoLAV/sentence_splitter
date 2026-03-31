from reformat_data import create_unique_dataset
from model_training import training
from prova import evaluate_flair, evaluate_nltk
import os

#CREATION OF THE UNIQUE FILES FOR TRAINING AND DEV, TO BE USED IN MODEL_TRAINING.PY
#structured in lists to be easily modified in case we want to add or remove files
all_train = [
    'sent_split_data/UD_English-EWT/en_ewt-ud-train.sent_split', 
    'sent_split_data/UD_English-GUM/en_gum-ud-train.sent_split',
    'sent_split_data/UD_English-ParTUT/en_partut-ud-train.sent_split',
    'sent_split_data/UD_Italian-ISDT/it_isdt-ud-train.sent_split',
    'sent_split_data/UD_Italian-MarkIT/it_markit-ud-train.sent_split',
    'sent_split_data/UD_Italian-ParTUT/it_partut-ud-train.sent_split',
    'sent_split_data/UD_Italian-VIT/it_vit-ud-train.sent_split'
]

all_dev = [
    'sent_split_data/UD_English-EWT/en_ewt-ud-dev.sent_split',
    'sent_split_data/UD_English-GUM/en_gum-ud-dev.sent_split',
    'sent_split_data/UD_English-ParTUT/en_partut-ud-dev.sent_split',
    'sent_split_data/UD_Italian-ISDT/it_isdt-ud-dev.sent_split',
    'sent_split_data/UD_Italian-MarkIT/it_markit-ud-dev.sent_split',
    'sent_split_data/UD_Italian-ParTUT/it_partut-ud-dev.sent_split',
    'sent_split_data/UD_Italian-VIT/it_vit-ud-dev.sent_split'
]

def main():

    #check if the reformatted data already exists, if not it create them
    if not os.path.exists('sent_split_data/unique_train.txt') or not os.path.exists('sent_split_data/unique_dev.txt'):
        print("Reformatting data...")
        create_unique_dataset(all_train, 'sent_split_data/unique_train.txt')
        create_unique_dataset(all_dev, 'sent_split_data/unique_dev.txt')

    #check if the model already exists, if not it trains it
    if not os.path.exists('final_model/best-model.pt') or not os.path.exists('final_model/final-model.pt'):
        print("Training model...")
        training()

    #TEST OF THE MODEL

     
if __name__ == '__main__':
    main()