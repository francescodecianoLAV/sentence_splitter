import os

os.environ['FLAIR_CACHE_ROOT'] = 'D:/flair_cache'

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

columns = {0: 'text', 1: 'label'}

corpus: Corpus = ColumnCorpus(
    data_folder='D:/sentence_splitter/sent_split_data', 
    column_format=columns,
    train_file='unique_train.txt',
    dev_file='unique_dev.txt',
    test_file='unique_test.txt'
)

label_type = 'label'
label_dict = corpus.make_label_dictionary(label_type=label_type)

#creation of the brain going foward and backward, to understand the context of the sentence
embedding_types = [
    FlairEmbeddings('multi-forward'),
    FlairEmbeddings('multi-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

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