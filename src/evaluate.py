import os
import nltk
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
import warnings
from reformat_data import create_unique_dataset

#just a little cleaning of the output, to avoid warnings from Flair and NLTK that are not relevant for the task
warnings.filterwarnings("ignore")

#download of the necessary resources for NLTK, with quiet=True to avoid unnecessary output
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# FUNCTION TO EVALUATE FLAIR-BASED MODEL
#creates a temporary file with the same format as the training and dev files, then evaluates the model on it
#and finally removes the temporary file, returning the F1 score, precision and recall
def evaluate_flair(file_input, tagger):
    temp_file = "temp_test_flair.txt"

    gold_labels = []
    pred_labels = []

    #creation of temporary test file for Flair, with all the files in input
    #in the same format as the training and dev files
    create_unique_dataset([file_input], temp_file, limit_words=100) 
    
    #creation of the corpus for evaluation, with the temporary file and the same column format as the training and dev files
    #test file is the temporary file created above
    corpus = ColumnCorpus('',
        {0: 'text', 1: 'label'},
        test_file=temp_file
    )

    #computes the evaluation of the model on the test set, with the gold label type 'label'(real answers)
    #and a mini batch size of 32 not to overload the memory
    #result = tagger.evaluate(corpus.test, gold_label_type='label', mini_batch_size=32)

    sentences = [sentence for sentence in corpus.test]   #creates a list of all the sentences

    for sentence in sentences:
        for token in sentence:
            true_label = token.get_label('label').value          #get the gold label so the true label, and convert it to 1 for EOS and 0 for O
            gold_labels.append(1 if true_label == 'EOS' else 0)  #add it to the list of gold labels

        tagger.predict(sentences, mini_batch_size=32)   #prediction of the model

        for token in sentence:
            pred_label = token.get_label(tagger.tag_type).value  #same for the predicted label, get it and convert it to 1 for EOS and 0 for O  
            pred_labels.append(1 if pred_label == 'EOS' else 0)  #add it to the list of predicted labels

    os.remove(temp_file) #removes the temporary file after evaluation

    #computation of true positives, false positives and false negatives for the calculation of precision and recall
    tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == 1 and p == 1)    #sum of cases where the gold label is 1 (EOS) and the predicted label is also 1 (EOS), so true positives
    fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == 0 and p == 1)    #sum of cases where the gold label is 0 (O) and the predicted label is 1 (EOS), so false positives
    fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == 1 and p == 0)    #sum of cases where the gold label is 1 (EOS) and the predicted label is 0 (O), so false negatives

    #calculation of precision, recall and F1 score, with checks to avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"f1": f1_score, "precision": precision, "recall": recall}

# FUNCTION TO EVALUATE NLTK
def evaluate_nltk(file_input, language):
    cleaned_words = []
    gold_labels = []
    
    #clean text from the input file, extracting the words and the gold labels, with the same format as the training
    #and dev files, so <EOS> for end of sentence and O for other words
    with open(file_input, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.split():
                if '<EOS>' in word:
                    cleaned_words.append(word.replace('<EOS>', ''))
                    gold_labels.append(1)                            #add 1 to the gold labels if the word is an end of sentence
                else:
                    cleaned_words.append(word)
                    gold_labels.append(0)                            #add 0 to the gold labels if the word is not an end of sentence
                    
    #reconstruction of the cleaned text, joining the cleaned words with a space, to be used for the tokenization with NLTK
    cleaned_text = " ".join(cleaned_words)
    
    language_nltk = 'italian' if language == 'it' else 'english'          #definition of language for NLTK
    tokenizer = nltk.data.load(f'tokenizers/punkt/{language_nltk}.pickle')  #load of the tokenizer for the specific language, it is able to understand the different rules of sentence splitting for different languages
    
    # span_tokenize return the start and end index of each sentence
    #this is useful to align with Flair that works on words
    spans_phrases = list(tokenizer.span_tokenize(cleaned_text))
    cut_chars = [span[1] for span in spans_phrases]      #get only the end index, so where it needs to cut the sentence and save them in a list
    
    pred_labels = [0] * len(cleaned_words)    #creates a list of zeros
    
    current_char_index = 0
    
    for i, word in enumerate(cleaned_words):    #enumerate to get both index and word
        current_char_index += len(word)
        
        #takes each cut index and checks if it is close to the current character index,
        #if it is it means that the model has predicted an end of sentence, so we set the predicted label to 1
        for cut in cut_chars:      
            if abs(current_char_index - cut) <= 1:       #tolerance if there is a fullstop, a space or other any punctuation 
                pred_labels[i] = 1
                break
                
        current_char_index += 1                         #for the space between words

    #computation of true positives, false positives and false negatives(same as above)
    tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == 1 and p == 0)

    #calculation of precision, recall and F1 score, with checks to avoid division by zero(same as above)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"f1": f1_score, "precision": precision, "recall": recall}

