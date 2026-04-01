from reformat_data import create_unique_dataset
from model_training import training
from evaluate import evaluate_flair, evaluate_nltk
import os
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus

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

#list of test files
all_test = [
    ('sent_split_data/UD_English-EWT/en_ewt-ud-test.sent_split', 'en'),
    ('sent_split_data/UD_English-GUM/en_gum-ud-test.sent_split', 'en'),
    ('sent_split_data/UD_English-ParTUT/en_partut-ud-test.sent_split', 'en'),
    ('sent_split_data/UD_English-PUD/en_pud-ud-test.sent_split', 'en'),
    ('sent_split_data/UD_Italian-ISDT/it_isdt-ud-test.sent_split', 'it'),
    ('sent_split_data/UD_Italian-MarkIT/it_markit-ud-test.sent_split', 'it'),
    ('sent_split_data/UD_Italian-ParTUT/it_partut-ud-test.sent_split', 'it'),
    ('sent_split_data/UD_Italian-VIT/it_vit-ud-test.sent_split', 'it')
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

    #EXECUTION OF TESTS
    print("LOADING OF FLAIR-BASED MODEL...")
    myModel = SequenceTagger.load('final_model/best-model.pt')     #load the best model trained

    print("\n" + "="*60)
    print(f"{'NAME OF DATASET':<40} | {'FLAIR':<7} | {'NLTK':<7}")
    print("="*60)

    for path, language in all_test:
        file_name = os.path.basename(path)    #get the name of the file from the path, to print it in the results table
        
        #get values of F1 score, precision and recall for both Flair and NLTK, with a try except to catch any error
        #that may occur during the evaluation and print it in the results table
        try:
            flair_results = evaluate_flair(path, myModel)
            nltk_results = evaluate_nltk(path, language)
            
            f1_flair = flair_results['f1']
            precision_flair = flair_results['precision']
            recall_flair = flair_results['recall']

            f1_nltk = nltk_results['f1']
            precision_nltk = nltk_results['precision']
            recall_nltk = nltk_results['recall']

            winnerF1 = "🏆" if f1_flair >= f1_nltk else " "
            winnerP = "🏆" if precision_flair >= precision_nltk else " "
            winnerR = "🏆" if recall_flair >= recall_nltk else " "
            
            print(f"{file_name:<40} F1| {f1_flair:.4f}{winnerF1}| {f1_nltk:.4f}")
            print(f"{file_name:<40} PRECISION| {precision_flair:.4f}{winnerP}| {precision_nltk:.4f}")
            print(f"{file_name:<40} RECALL| {recall_flair:.4f}{winnerR}| {recall_nltk:.4f}")

        except Exception as e:
            print(f"{file_name:<40} | ERROR IN TEST: {e}")

    print("="*60)


if __name__ == '__main__':
    main()