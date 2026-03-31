import os

#the code is built for my PC, so I set the cache root to a specific folder, but it can be changed to any other folder
os.environ['FLAIR_CACHE_ROOT'] = 'D:/flair_cache' 

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

def training():
    #defition of the columns of dataset, first the word, then the label : EOS for end of sentence, O for other words
    columns = {0: 'text', 1: 'label'}

    #creation of the corpus, with the train and dev files and the column format 
    corpus: Corpus = ColumnCorpus(
        data_folder='D:/sentence_splitter/sent_split_data', 
        column_format=columns,
        train_file='unique_train.txt',
        dev_file='unique_dev.txt'
    )

    #creation of the label dictionary, with the type of label that it has to predict
    #it scans the corpus and extracts the unique labels, in this case EOS and O
    label_type = 'label'
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    #creation of the brain going foward and backward, to understand the context of the sentence
    #these lines create the embeddings
    #multi means that is able to understand different languages
    embedding_types = [
        FlairEmbeddings('multi-forward'),
        FlairEmbeddings('multi-backward'),
    ]

    #union of the two brains
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    #creation of the neural network, with the embeddings, the label dictionary and the type of label to predict
    print("building the neural network...")
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,        
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type=label_type,
        use_crf=True 
    )

    print("start training!")
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        'final_model', 
        learning_rate=0.1,
        mini_batch_size=16,
        max_epochs=30,
        embeddings_storage_mode='none'
    )