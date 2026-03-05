FROM nvcr.io/nvidia/pytorch:22.08-py3

WORKDIR /workspace/animal2vec

# Versione pip richiesta da animal2vec
RUN pip install --upgrade pip==24.0

# Rimuove fisicamente llvmlite preinstallata come distutils
RUN rm -rf /opt/conda/lib/python3.8/site-packages/llvmlite* /opt/conda/lib/python3.8/site-packages/llvmlite-*.dist-info /opt/conda/lib/python3.8/site-packages/llvmlite-*.egg-info

# Installa le dipendenze
RUN pip install -r requirements.txt

# Installa fairseq dal commit specifico (sovrascrive la versione in requirements.txt)
RUN pip install git+https://github.com/facebookresearch/fairseq.git@920a548ca770fb1a951f7f4289b4d3a0c1bc226f

# nonostante sia nei requirements viene sovrascritto da fairseq a 4.67.3 e ha un bug con Il bug 'sp' attribute è stato introdotto in tqdm 4.66.x e fixato in versioni successive.
RUN pip install tqdm==4.64.1


# Necessario per "import nn" che animal2vec usa internamente
ENV PYTHONPATH="/workspace/animal2vec:"


# Copia i file del repo nell'immagine
COPY . .