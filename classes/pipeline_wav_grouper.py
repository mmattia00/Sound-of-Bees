import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

class WAVPipelineGrouper:
    """
    Raggruppa file WAV per peaktime simile ed estrae i metadati dai nomi dei file.
    Seleziona il canale con HNR più forte per ogni gruppo peaktime.
    """
    
    def __init__(self, folder_path, peaktime_tolerance=0.02):
        """
        Args:
            folder_path (str): Percorso della cartella contenente i file WAV
            peaktime_tolerance (float): Tolleranza per raggruppare peaktime simili (in secondi)
        """
        self.folder_path = Path(folder_path)
        self.peaktime_tolerance = peaktime_tolerance
        self.files_metadata = []
        self.groups = defaultdict(list)
        
    def extract_metadata(self, filename):
        """
        Estrae metadati dal nome del file usando regex.
        Formato atteso: ch_XX_peaktime_Y.YYY_windowstart_Z.ZZZ_windowend_W.WWW_hnrvalue_V.VV.wav
        
        Returns:
            dict con chiavi: 'filename', 'channel', 'peaktime', 'windowstart', 
                           'windowend', 'hnrvalue'
        """
        # sostituisci .wav com ""
        filename = filename.replace('.wav', '')
        pattern = r'ch_(\d+)_peaktime_([\d.]+)_windowstart_([\d.]+)_windowend_([\d.]+)_hnrvalue_(-?[\d.]+)'
        match = re.match(pattern, filename)
        
        if not match:
            print(f"⚠️  Impossibile parsare: {filename}")
            return None
        
        return {
            'filename': filename,
            'channel': int(match.group(1)),
            'peaktime': float(match.group(2)),
            'windowstart': float(match.group(3)),
            'windowend': float(match.group(4)),
            'hnrvalue': float(match.group(5))
        }
    
    def load_files(self):
        """Carica i metadati di tutti i file WAV dalla cartella."""
        wav_files = sorted(self.folder_path.glob('*.wav'))
        
        if not wav_files:
            print(f"❌ Nessun file WAV trovato in {self.folder_path}")
            return False
        
        print(f"📁 Trovati {len(wav_files)} file WAV")
        
        for wav_file in wav_files:
            metadata = self.extract_metadata(wav_file.name)
            if metadata:
                self.files_metadata.append(metadata)
        
        print(f"✅ Estratti metadati da {len(self.files_metadata)} file")
        return True
    
    def group_by_peaktime(self):
        """
        Raggruppa i file per peaktime simile.
        Usa un approccio iterativo: il primo file di un gruppo definisce il riferimento.
        """
        if not self.files_metadata:
            print("❌ Nessun metadata disponibile. Eseguire load_files() prima.")
            return
        
        # Ordina per peaktime
        sorted_files = sorted(self.files_metadata, key=lambda x: x['peaktime'])
        assigned = set()
        group_counter = 0
        
        for i, file in enumerate(sorted_files):
            if i in assigned:
                continue
            
            # Questo file inzia un nuovo gruppo
            reference_peaktime = file['peaktime']
            group = [file]
            assigned.add(i)
            
            # Trova tutti gli altri file simili
            for j in range(i + 1, len(sorted_files)):
                if j in assigned:
                    continue
                
                if abs(sorted_files[j]['peaktime'] - reference_peaktime) <= self.peaktime_tolerance:
                    group.append(sorted_files[j])
                    assigned.add(j)
                else:
                    # Non troveremo più file simili perché è ordinato
                    break
            
            # Salva il gruppo
            self.groups[group_counter] = group
            group_counter += 1
        
        print(f"📊 {group_counter} gruppi creati con tolleranza ±{self.peaktime_tolerance}s")
    
    def get_best_channels(self):
        """
        Per ogni gruppo peaktime, ritorna il file con HNR value più forte.
        
        Returns:
            list: Lista di dizionari con i file selezionati (uno per gruppo)
        """
        if not self.groups:
            print("❌ Nessun gruppo disponibile. Eseguire group_by_peaktime() prima.")
            return []
        
        best_files = []
        
        for group_id, group_files in sorted(self.groups.items()):
            # HNR value più alto = segnale più pulito
            best_file = max(group_files, key=lambda x: x['hnrvalue'])
            best_files.append({
                'group_id': group_id,
                'group_size': len(group_files),
                'peaktime': best_file['peaktime'],
                'channel': best_file['channel'],
                'hnrvalue': best_file['hnrvalue'],
                'filename': best_file['filename'],
                'all_channels_in_group': [f['channel'] for f in group_files],
                'all_hnr_values': {f['channel']: f['hnrvalue'] for f in group_files}
            })
        
        return best_files
    
    def print_summary(self):
        """Stampa un sommario dei gruppi e dei file selezionati."""
        best_files = self.get_best_channels()
        
        if not best_files:
            return
        
        print("\n" + "="*80)
        print("📋 SOMMARIO RAGGRUPPAMENTO")
        print("="*80)
        print(f"{'Gruppo':<8} {'Peaktime':<12} {'Canali':<25} {'Ch Scelto':<12} {'HNR':<10}")
        print("-"*80)
        
        for item in best_files:
            all_ch = sorted(item['all_channels_in_group'])
            ch_str = f"[{', '.join(map(str, all_ch))}]"
            print(f"{item['group_id']:<8} {item['peaktime']:<12.3f} {ch_str:<25} {item['channel']:<12} {item['hnrvalue']:<10.2f}")
        
        print("-"*80)
        print(f"✅ File da processare: {len(best_files)} (riduzione da {len(self.files_metadata)} file)")
        print("="*80 + "\n")
        
        return best_files
    
    def get_processing_list(self):
        """Ritorna la lista di file da processare con la pipeline."""
        best_files = self.get_best_channels()
        # Ritorna solo il nome del file .wav, non il percorso completo
        file_paths = [item['filename'] for item in best_files]
        return file_paths, best_files



# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    # CONFIGURAZIONE
    FOLDER_PATH = "/path/to/your/wav/folder"  # 👈 MODIFICA QUESTO
    PEAKTIME_TOLERANCE = 0.02  # ±0.02 secondi (puoi usare 0.01 per più stretto)
    
    # Inizializza il grouper
    grouper = WAVPipelineGrouper(FOLDER_PATH, peaktime_tolerance=PEAKTIME_TOLERANCE)
    
    # Carica i file e estrai metadati
    if grouper.load_files():
        # Raggruppa per peaktime simile
        grouper.group_by_peaktime()
        
        # Stampa sommario
        best_files = grouper.print_summary()
        
        # Ottieni la lista di file da processare
        file_paths, metadata = grouper.get_processing_list()
        
        print("📁 File da processare con la pipeline:")
        for path, meta in zip(file_paths, metadata):
            print(f"  • {path}")
            print(f"    → Canale {meta['channel']}, Peaktime {meta['peaktime']:.3f}s, HNR {meta['hnrvalue']:.2f}")
        
        # ====================================================================
        # INTEGRAZIONE NELLA TUA PIPELINE
        # ====================================================================
        # Ora puoi usare file_paths nella tua pipeline:
        #
        # for file_path in file_paths:
        #     result = tua_pipeline(file_path)
        #     # processa il risultato
        #
        # Oppure se usi il metadata per logging:
        #
        # for file_path, meta in zip(file_paths, metadata):
        #     result = tua_pipeline(file_path)
        #     print(f"Processato gruppo {meta['group_id']}: {file_path}")

