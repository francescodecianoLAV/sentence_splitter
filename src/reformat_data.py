import os

def create_unique_dataset(list_file_input, file_output, limit_words=100):
    print(f"Creation unique-file {file_output} ...")
    
    with open(file_output, 'w', encoding='utf-8') as f_out:
        for file_input in list_file_input:
            print(f"working on: {file_input}...")
            
            if not os.path.exists(file_input):
                print(f" File NOT FOUND! ({file_input}).")
                continue

            words_count = 0
            next = True

            with open(file_input, 'r', encoding='utf-8') as f_in:
                lines = f_in.readlines()
                for line in lines:
                    words = line.split()
                    
                    # 1. LA MODIFICA CHIAVE: Ignoriamo completamente gli "A capo" del professore!
                    # Fonderemo tutto il testo in un unico flusso ininterrotto.
                    if len(words) == 0:
                        continue 
                    
                    for word in words:
                        is_eos = False 
                        
                        if '<EOS>' in word:
                            word_cleaned = word.replace('<EOS>', '')
                            label = 'EOS'
                            is_eos = True
                        else:
                            word_cleaned = word
                            label = 'O'
                            
                        if word_cleaned.strip():
                            f_out.write(f"{word_cleaned}\t{label}\n")
                            words_count += 1
                            next = False
                        
                        # 2. IL TAGLIO SALVA-GPU: Andiamo a capo SOLO quando raggiungiamo le 100 parole
                        if is_eos and words_count >= limit_words:
                            f_out.write("\n") 
                            words_count = 0
                            next = True
            
            if not next:
                f_out.write("\n")
            
    print(f"Finish! SUCCESS: {file_output}\n")







# 1. TRAINING FILES
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

# 3. TEST FILES
all_test = [
    'sent_split_data/UD_English-EWT/en_ewt-ud-test.sent_split',
    'sent_split_data/UD_English-GUM/en_gum-ud-test.sent_split',
    'sent_split_data/UD_English-ParTUT/en_partut-ud-test.sent_split',
    'sent_split_data/UD_English-PUD/en_pud-ud-test.sent_split',
    'sent_split_data/UD_Italian-ISDT/it_isdt-ud-test.sent_split',
    'sent_split_data/UD_Italian-MarkIT/it_markit-ud-test.sent_split',
    'sent_split_data/UD_Italian-ParTUT/it_partut-ud-test.sent_split',
    'sent_split_data/UD_Italian-VIT/it_vit-ud-test.sent_split'
]

create_unique_dataset(all_test, 'sent_split_data/unique_test.txt')


