import os

#function to create a unique file with all the data, with the format "word \t label" and a new line every 100 words
#  or at the end of the sentence (EOS)
#limit_words is used to avoid GPU memory to be overloaded
def create_unique_dataset(list_file_input, file_output, limit_words=100):  
    print(f"Creation unique-file {file_output} ...")
    
    #open the output file to write on
    with open(file_output, 'w', encoding='utf-8') as f_out:
        #iterate over list of input files
        for file_input in list_file_input:
            print(f"working on: {file_input}...")
            
            #check if the file exists
            if not os.path.exists(file_input):
                print(f" File NOT FOUND! ({file_input}).")
                continue

            words_count = 0     #counter of the words in the current sentence, to avoid GPU overload
            next = True         #a flag to check if we have put a new line at the end of the sentence, to avoid putting multiple new lines in a row

            #open the input file to read from, iterate over the lines and split them into words
            with open(file_input, 'r', encoding='utf-8') as f_in:
                lines = f_in.readlines()
                for line in lines:
                    words = line.split()
                    
                    if len(words) == 0:   #if the line is empty, for example at the end of a paragraph,
                        continue          #we skip it, forcing Flair to understand by itself that we have reached the end of a sentence
                    
                    for word in words:     #iteration over words in the line
                        is_eos = False     #flag to check if we have reached the end of a sentence
                        
                        #if the word contains the tag <EOS>, we remove it and we label the word as EOS, otherwise we label it as O (other)
                        if '<EOS>' in word: 
                            word_cleaned = word.replace('<EOS>', '')     #replace the tag with an empty string
                            label = 'EOS'                                
                            is_eos = True                                #set the flag to true, to indicate that we have reached the end of a sentence
                        else:
                            word_cleaned = word                          
                            label = 'O'
                        
                        #if the word_cleaned contains something different from spaces
                        if word_cleaned.strip():
                            f_out.write(f"{word_cleaned}\t{label}\n")        #write the word and the label in the output file, separated by a tab
                            words_count += 1                                 #increment the counter of words in the current sentence
                            next = False                                     #set the flag to false, to indicate that we have not put a new line at the end of the sentence yet
                        
                        #if we have reached the end of a sentence and we have reached the limit of words,
                        #we put a new line in the output file, to indicate to Flair that we have reached the end of a sentence 
                        #and to avoid GPU overload
                        if is_eos and words_count >= limit_words:
                            f_out.write("\n") 
                            words_count = 0                     #reset counter of words in the current sentence
                            next = True                         #we put a new line
            
            #if to check if we have put a new line between files to separate them
            if not next:
                f_out.write("\n")
            
    print(f"Finish! SUCCESS: {file_output}\n")


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
create_unique_dataset(all_train, 'sent_split_data/unique_train.txt')

# 2. DEV FILES 
all_dev = [
    'sent_split_data/UD_English-EWT/en_ewt-ud-dev.sent_split',
    'sent_split_data/UD_English-GUM/en_gum-ud-dev.sent_split',
    'sent_split_data/UD_English-ParTUT/en_partut-ud-dev.sent_split',
    'sent_split_data/UD_Italian-ISDT/it_isdt-ud-dev.sent_split',
    'sent_split_data/UD_Italian-MarkIT/it_markit-ud-dev.sent_split',
    'sent_split_data/UD_Italian-ParTUT/it_partut-ud-dev.sent_split',
    'sent_split_data/UD_Italian-VIT/it_vit-ud-dev.sent_split'
]
create_unique_dataset(all_dev, 'sent_split_data/unique_dev.txt')



