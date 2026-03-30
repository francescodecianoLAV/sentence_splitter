import os
import nltk
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
import warnings

# Nascondiamo i fastidiosi warning rossi
warnings.filterwarnings("ignore")

# Scarica i modelli linguistici di NLTK (lo fa solo la prima volta)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------
# FUNZIONE 1: VALUTA IL TUO MODELLO FLAIR
# ---------------------------------------------------------
def valuta_flair(file_input, tagger):
    temp_file = "temp_test_flair.txt"
    
    # Prepariamo il file coi tagli di sicurezza ogni 100 parole
    with open(file_input, 'r', encoding='utf-8') as f_in, open(temp_file, 'w', encoding='utf-8') as f_out:
        words_count = 0
        for linea in f_in:
            if not linea.strip(): continue
            for parola in linea.strip().split():
                is_eos = False
                if "<EOS>" in parola:
                    parola_pulita = parola.replace('<EOS>', '')
                    f_out.write(f"{parola_pulita}\tEOS\n")
                    is_eos = True
                else:
                    f_out.write(f"{parola}\tO\n")
                
                words_count += 1
                
                # IL TAGLIO SALVA-GPU: Svuotiamo la memoria dopo 100 parole
                if is_eos and words_count >= 100:
                    f_out.write("\n")
                    words_count = 0
                    
        if words_count > 0:
            f_out.write("\n")

    # Facciamo valutare a Flair (con mini_batch_size basso per non fonderlo!)
    corpus = ColumnCorpus('', {0: 'text', 1: 'label'}, test_file=temp_file)
    result = tagger.evaluate(corpus.test, gold_label_type='label', mini_batch_size=2)
    
    os.remove(temp_file) # Pulizia
    return result.main_score # Restituisce l'F-Score

# ---------------------------------------------------------
# FUNZIONE 2: VALUTA NLTK
# ---------------------------------------------------------
# ---------------------------------------------------------
# FUNZIONE 2: VALUTA NLTK (A prova di Deriva dell'Allineamento)
# ---------------------------------------------------------
def valuta_nltk(file_input, lingua):
    parole_pulite = []
    gold_labels = []
    
    # 1. Estraiamo le parole e i tag
    with open(file_input, 'r', encoding='utf-8') as f:
        for linea in f:
            for parola in linea.split():
                if '<EOS>' in parola:
                    parole_pulite.append(parola.replace('<EOS>', ''))
                    gold_labels.append(1)
                else:
                    parole_pulite.append(parola)
                    gold_labels.append(0)
                    
    # Ricostruiamo il testo in un'unica stringa pulita
    testo_pulito = " ".join(parole_pulite)
    
    # 2. Tokenizzazione avanzata (usiamo gli indici dei caratteri, non le parole!)
    lingua_nltk = 'italian' if lingua == 'it' else 'english'
    tokenizer = nltk.data.load(f'tokenizers/punkt/{lingua_nltk}.pickle')
    
    # span_tokenize restituisce (inizio, fine) di ogni frase in base ai caratteri
    spans_frasi = list(tokenizer.span_tokenize(testo_pulito))
    tagli_caratteri = [span[1] for span in spans_frasi] # Prendiamo solo l'indice finale
    
    pred_labels = [0] * len(parole_pulite)
    
    # 3. Allineamento chirurgico
    indice_carattere_attuale = 0
    
    for i, parola in enumerate(parole_pulite):
        # Avanziamo il contatore dei caratteri per la lunghezza della parola
        indice_carattere_attuale += len(parola)
        
        # Controlliamo se il taglio di NLTK cade esattamente alla fine di questa parola
        # Usiamo una tolleranza di 1 carattere per via dello spazio vuoto (" ")
        for taglio in tagli_caratteri:
            if abs(indice_carattere_attuale - taglio) <= 1:
                pred_labels[i] = 1
                break
                
        # Aggiungiamo 1 per lo spazio che separa questa parola dalla successiva
        indice_carattere_attuale += 1

    # 4. Calcolo rigoroso dell'F-Score
    tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1

# ---------------------------------------------------------
# ESECUZIONE DEL TEST
# ---------------------------------------------------------
print("Caricamento del tuo modello Flair...")
mio_modello = SequenceTagger.load('final_model/best-model.pt')

# INSERISCI QUI LA LISTA DEI FILE CHE VUOI TESTARE
file_da_testare = [
    ('sent_split_data/UD_English-EWT/en_ewt-ud-test.sent_split', 'en'),
    ('sent_split_data/UD_English-GUM/en_gum-ud-test.sent_split', 'en'),
    ('sent_split_data/UD_English-ParTUT/en_partut-ud-test.sent_split', 'en'),
    ('sent_split_data/UD_English-PUD/en_pud-ud-test.sent_split', 'en'),
    ('sent_split_data/UD_Italian-ISDT/it_isdt-ud-test.sent_split', 'it'),
    ('sent_split_data/UD_Italian-MarkIT/it_markit-ud-test.sent_split', 'it'),
    ('sent_split_data/UD_Italian-ParTUT/it_partut-ud-test.sent_split', 'it'),
    ('sent_split_data/UD_Italian-VIT/it_vit-ud-test.sent_split', 'it')
    
]

print("\n" + "="*60)
print(f"{'NOME DEL DATASET':<40} | {'FLAIR':<7} | {'NLTK':<7}")
print("="*60)

for percorso, lingua in file_da_testare:
    nome_file = os.path.basename(percorso)
    
    try:
        f_flair = valuta_flair(percorso, mio_modello)
        f_nltk = valuta_nltk(percorso, lingua)
        
        # Scegliamo chi ha vinto
        vincitore = "🏆" if f_flair > f_nltk else " "
        
        print(f"{nome_file:<40} | {f_flair:.4f}{vincitore}| {f_nltk:.4f}")
    except Exception as e:
        print(f"{nome_file:<40} | ERRORE NEL TEST: {e}")

print("="*60)
print("Il simbolo 🏆 indica che il tuo modello ha battuto NLTK.")