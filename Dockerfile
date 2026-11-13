# Image du travailleur de modération pour RunPod Serverless (lot 6, 13/11/2026).
#
# Construite par RunPod depuis le dépôt GitHub, jamais sur un poste ni sur le VPS
# (exception à « pas de Docker » acceptée le 13/11/2026). Même base que ocr-worker.
#
# Aucun secret ici : la clé privée du travailleur (WORKER_PRIVATE_KEY) et sa référence
# sont des variables de l'endpoint, la clé venant d'un secret RunPod.
FROM runpod/base:1.4.1-cuda1290-ubuntu2404

# Tesseract (français et anglais), ZBar pour les QR codes, FFmpeg pour la vidéo.
# Ubuntu 24.04 nomme la bibliothèque ZBar « libzbar0t64 ».
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng libzbar0t64 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Le venv isole le travailleur des paquets Python de l'image (même choix que ocr-worker).
# HF_HOME : le modèle Whisper est mis en cache à la construction et relu au même endroit.
ENV UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/huggingface \
    TMP_DIR=/tmp/howe-moderation-worker \
    CAPABILITIES=image,video,asr \
    SLOTS=image=3,video=1,asr=1 \
    WHISPER_MODEL=small \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float16 \
    IDLE_EXIT_SECONDS=20 \
    MAX_RUN_SECONDS=1800

RUN uv venv /opt/venv --python 3.12

WORKDIR /app
COPY requirements.txt requirements-asr.txt requirements-runpod.txt ./
RUN uv pip install --no-cache -r requirements-runpod.txt

# CTranslate2 cherche cuBLAS et cuDNN par le chargeur de bibliothèques : on lui montre
# celles installées par pip dans le venv.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib

COPY howe_worker/ howe_worker/
COPY handler.py .

# Whisper small dans l'image : un conteneur neuf ne télécharge rien, GPU allumé. Le
# conteneur de construction n'a pas de GPU : le modèle se charge ici sur processeur,
# les poids sont les mêmes. L'étape vérifie aussi que Tesseract, ZBar, FFmpeg et le
# paquet du travailleur s'importent : une bibliothèque manquante fait échouer la
# construction, pas le premier appel.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" \
 && python -c "import howe_worker.serverless, pyzbar.pyzbar, cv2; print('travailleur', __import__('howe_worker').VERSION)" \
 && tesseract --list-langs | grep -q fra \
 && ffmpeg -hide_banner -version | head -n1

CMD ["python", "-u", "handler.py"]
