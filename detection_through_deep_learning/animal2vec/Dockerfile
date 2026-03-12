FROM nvcr.io/nvidia/pytorch:22.08-py3

WORKDIR /workspace/animal2vec

RUN pip install --upgrade pip==24.0

RUN rm -rf /opt/conda/lib/python3.8/site-packages/llvmlite* \
           /opt/conda/lib/python3.8/site-packages/llvmlite-*.dist-info \
           /opt/conda/lib/python3.8/site-packages/llvmlite-*.egg-info

# Rimuovi fairseq e apex preinstallate dall'immagine base
RUN pip uninstall fairseq apex -y || true

# Copia SOLO requirements.txt prima di installare
COPY requirements.txt .

# Installa dipendenze (fairseq==0.12.2 va rimossa da requirements.txt)
RUN pip install -r requirements.txt

# Installa fairseq dal commit corretto su ambiente pulito
RUN pip install --no-build-isolation \
    git+https://github.com/facebookresearch/fairseq.git@920a548ca770fb1a951f7f4289b4d3a0c1bc226f

RUN pip install tqdm==4.64.1

ENV PYTHONPATH="/workspace/animal2vec:"

COPY . .
