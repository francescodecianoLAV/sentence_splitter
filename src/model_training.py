import os

#the code is built for my PC, so I set the cache root to a specific folder, but it can be changed to any other folder
#os.environ['FLAIR_CACHE_ROOT'] = 'D:/flair_cache' 

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
        hidden_size=256,                 #number of neurons in the hidden layer, can be increased(also the number of layers adding rnn_layers= 2 for example)   
        embeddings=embeddings,           #use embeddings defined above
        tag_dictionary=label_dict,       #use the label dictionary created above
        tag_type=label_type,             #type of label to predict
        use_crf=True                     #use CRF to improve the performance of the model, it helps to understand the dependencies between labels, in this case it helps to understand that after an EOS there is an O and viceversa
    )

    print("start training!")
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)    #creation of the trainer, with the model and the corpus

    trainer.train(                          #creates the final_model after the last epoch and when it finds it, also the best model
        'final_model',                      #folder where the model will be saved, it will create a folder with the name of the model and save the best and final model inside
        learning_rate=0.1,                  #learning rate, can be increased but it can cause the model to diverge, it is better to keep it low to avoid this problem
        mini_batch_size=16,                 #size of the batch, can be increased but it requires more memory, it is better to keep it low to avoid out of memory error
        max_epochs=30,                      #30 epochs(could also increased)
        embeddings_storage_mode='none'      #avoids the network to remember values of the embeddings of the previous batches(saves memory)
    )