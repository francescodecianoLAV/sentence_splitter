# Sentence splitter
The aim of the project is to create a sentence splitter able to beat NLTK.
To create the sentence splitter, the library Flair of python has been chosen; its ecosystem is based on Pytorch and it's like a more powerful version of the embeddings.
Flair reads each character of a sentence, forward and backward, and in the end compare the two analysis of the sentence, becoming able to understand the meaning of a word based on the context and also to recognize out of dictionary words; it makes no difference what language is used(english or italian only cause they are the language of training and dev files).

The project is organized in four files:

```python
reformat_data.py 
```
This file contains the function to write the input files in a two coloumns format, to allow Flair to analyze them.

```python
model_training.py
```
It contains the training of the Flair-based model; at the end of the process two models will be created: the final one and best one.

```python
evaluate.py
```
This files contains the two functions for the evaluation of test files, one for the trained model and one for nltk; both of them return a dictonary with F1, precision and recall.

```python
main.py
```
This file manages the execution: it creates the file.txt if they are not already created, trains the model if is not already present and at the evaluates and compare the results.

Training is expensive and it lasts a long time; I tried my best to reduce it at minimum without losing performance, even using my personal gaming pc for the operation.
For evaluation I suggest to use Google Colab or execute it on graphic card.

On your pc run only 
```python
main.py
```
For Colab upload the project on your Google Drive and use a file like the following one 
[https://colab.research.google.com/drive/1byEvib3CASi96HPur5DOyrdvXqHqrd6Q?usp=sharing](https://colab.research.google.com/drive/1byEvib3CASi96HPur5DOyrdvXqHqrd6Q?usp=sharing)