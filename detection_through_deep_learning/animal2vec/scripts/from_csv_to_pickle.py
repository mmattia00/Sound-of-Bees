import pandas as pd
from datetime import timedelta

# --- Carica il tuo CSV ---
csv_path = 'E:/training_dataset_finetuning_raw/pickle_file_preparation.csv'
# Adatta sep='\\s+' se è space-separated, oppure sep=',' se è comma-separated
# Leggi con header=0 (default): usa la prima riga come nomi colonne
df_raw = pd.read_csv(csv_path, sep=',')

# Stampa le colonne reali per vedere i nomi esatti
print(df_raw.columns.tolist())
print(df_raw.head())

# --- Costruisci il DataFrame nel formato atteso da animal2vec ---
df_labels = pd.DataFrame({
    'Name':          'stop_signal_candidate',                          # Il nome della tua classe
    'AudioFile':     df_raw['filename'],
    'StartRelative': df_raw['onset_sec'].apply(lambda s: timedelta(seconds=float(s))),
    'EndRelative':   df_raw['offset_sec'].apply(lambda s: timedelta(seconds=float(s))),
    'Focal':         False                             # True se hai info focal
})

# --- Salva come pickle ---
df_labels.to_pickle('stop_signal_labels.pkl')

print(df_labels.head())
print(f"\nTotale eventi: {len(df_labels)}")