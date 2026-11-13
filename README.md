# Travailleur de modération de Howé

Il lit les photos des annonces pour la modération automatique de Howé (spécification « Modération automatisée des annonces de Howé », lot 3b). Il ne décide jamais : il prend une tâche chez Howé, lit le fichier par un lien signé, et rend ce qu'il a lu. Howé applique ses règles et tranche.

Ni Docker ni MinIO (décision du 12/11/2026) : un environnement virtuel Python et un service systemd.

## Ce qu'il fait (`photo_check`)

1. Il télécharge l'original par le lien signé de la tâche (15 minutes), taille plafonnée.
2. Il vérifie le vrai type du fichier (signature) et ses dimensions avant tout décodage. Une image de plus de 40 mégapixels est refusée, et une photo de plus de 3 840 px est réduite.
3. Tesseract (français et anglais) lit l'image deux fois : telle quelle, puis en niveaux de gris contrastés, agrandie si elle est petite. La lecture se fait dans un sous-processus limité en mémoire et en temps, et tué si Howé annule la tâche.
4. ZBar lit les QR codes ; OpenCV relit ceux que ZBar a manqués.
5. Il rend les lignes lues avec leur confiance, et les QR codes. Le dossier temporaire de la tâche est effacé, même en cas d'échec. Le journal (JSON) ne contient jamais le texte lu.

## La vidéo (version 0.2.0)

Howé ne confie ses tâches vidéo qu'à un travailleur en 0.2.0 ou plus.

- **`video_prepare`** (compétence `video`) :
  1. il vérifie le vrai type du fichier, puis la durée avec ffprobe (fichiers locaux et formats vidéo seulement). Une vidéo de plus de 45,5 s, sans image ou illisible est refusée ;
  2. il encode une fois la copie publiable : H.264 High, 1 280 px, CRF 23, 2,5 Mbit/s au plus, AAC, faststart, sans métadonnées ;
  3. il produit l'aperçu (WebP à 1 s), le son (WAV mono 16 kHz) et les images (deux par seconde, plus chaque changement de plan, doublons écartés par dHash) ;
  4. il dépose tout chez Howé par l'API signée.
- **`video_frames_read`** (compétence `image`) : il lit chaque image comme une photo (Tesseract et QR codes), avec son instant.
- **`video_transcribe`** (compétence `asr`) : Faster-Whisper small, int8 sur processeur, chargé une seule fois. Il faut environ 800 Mo de mémoire. Le modèle n'est jamais téléchargé pendant une tâche ; mettre `WHISPER_DOWNLOAD=1` pour le premier démarrage, s'il n'est pas déjà dans le cache.

## Les mesures (version 0.3.0)

Avec chaque résultat ou échec, le travailleur rend des nombres, jamais du texte. Howé les garde avec la tâche et les montre sur l'écran « Modération automatique » ; ils servent à dimensionner les serveurs sans s'y connecter.

- `duration_ms` : durée de la tâche ;
- `tools_memory_mb` : mémoire maximale d'un programme lancé pour la tâche (Tesseract, FFmpeg), lue exactement à sa fin ;
- `worker_memory_mb` : plus haut niveau de mémoire du travailleur depuis son démarrage, Whisper compris. Le système ne la donne pas tâche par tâche ;
- `frames_read` : images lues, pour la lecture d'une vidéo.

Les mêmes nombres figurent dans la ligne `task.finished` du journal.

## Le journal (version 0.4.1)

Une ligne JSON par événement, sur la sortie standard : `journalctl -u howe-moderation-worker` sur un serveur, onglet « Logs » de l'endpoint sur RunPod. Chaque ligne porte `ts`, `level` et `event` ; pendant une tâche, aussi `task` (sa référence) et `kind`.

**`LOG_LEVEL=info`** (par défaut) : la vie du travailleur et de chaque tâche.

| Événement | Quand |
|---|---|
| `worker.started`, `worker.woken` | démarrage (serveur) ou réveil (RunPod) : version, hôte de Howé, début de la référence, compétences, places, Whisper |
| `worker.container_started`, `worker.gpu`, `worker.warmed` | RunPod : conteneur lancé, GPU vus, Whisper chargé et en combien de temps |
| `tasks.leased` | des tâches prises : combien, de quels types |
| `task.started`, `task.finished` | début et fin d'une tâche : issue, code d'échec, durée, mémoire |
| `task.cancelled_by_howe` | Howé a annulé la tâche (annonce retirée, nouvelle version…) |
| `worker.waiting` | toutes les 10 minutes sans tâche : le travailleur est vivant |
| `worker.idle_exit`, `worker.wake_done`, `worker.stopped` | arrêt, avec le bilan des tâches et la durée |

Toujours écrits, quel que soit le niveau : `warning` (Howé injoignable, battement de cœur manqué, résultat non rendu, tâche en échec) et `error` (travailleur refusé, avec ce qu'il faut vérifier ; erreur de configuration ; `task.crashed`, avec le type de l'erreur et l'endroit du code où elle est née).

**`LOG_LEVEL=debug`** : en plus, chaque appel à Howé (`howe.call` : méthode, chemin, statut, durée), chaque fichier lu ou déposé (`file.download`, `file.upload` : nom, taille, durée) et chaque étape d'une tâche (`step.done` / `step.failed` : ouverture, lectures Tesseract, QR codes, sondage, encodage, aperçu, son, images, transcription…, avec leur durée et des nombres). Pour chercher une panne ou une lenteur, puis revenir à `info` : une vidéo écrit plus de 100 lignes.

**Ce qui n'est jamais écrit** : le texte lu, les paroles transcrites, le contenu des QR codes, les liens signés (ni leur adresse ni leurs paramètres), le jeton du bail, la clé privée. Le message d'une erreur inattendue non plus, car il peut contenir une adresse : seulement son type et l'endroit.

## Spécialiser un serveur

Un serveur ne prend que ce qu'il annonce (`CAPABILITIES`) et ce que Howé lui a permis à l'inscription (`--capabilities`). Un serveur modeste : `image`. Un serveur avec plus de mémoire : `video,asr` (ou les trois). C'est le seul réglage à changer ; ajouter un serveur ne touche pas Howé.

## Sur RunPod Serverless (version 0.4.0, lot 6)

Choix du 13/11/2026 : le travailleur tourne sur RunPod, pour que la modération ne prenne jamais de processeur au site. RunPod construit l'image depuis ce dépôt GitHub (le `Dockerfile` à la racine) et ne fait tourner un conteneur que lorsque Howé le réveille : on ne paie que le temps de calcul.

**Le principe.** Quand des tâches attendent, Howé appelle `/run` sur l'endpoint, avec une entrée vide : rien de l'annonce ne passe par RunPod. Le conteneur démarre (`handler.py`), prend ses tâches par l'API signée de Howé comme n'importe quel travailleur, les traite, et s'arrête quand il n'a plus rien eu depuis `IDLE_EXIT_SECONDS` (20 s). RunPod coupe alors le conteneur. Howé n'en réveille jamais plus que `MODERATION_RUNPOD_MAX_CONTAINERS` (2) à la fois.

**Mise en place, une fois :**

1. **Le dépôt GitHub.** Le `.gitignore` est déjà là ; il doit l'être **avant** le premier commit (un `.gitignore` arrivé après ne retire rien de l'historique) :
   ```bash
   cd ~/Dev/howe-moderation-worker
   git init && git add . && git status   # vérifier : ni venv/, ni clé, ni image
   git commit -m "Travailleur de modération : RunPod Serverless"
   ```
   Puis le pousser sur un dépôt **privé** GitHub.
2. **Le travailleur dans Howé** : `php artisan moderation:worker-add runpod --capabilities=image,video,asr`. Garder la référence et la clé privée affichées.
3. **Le secret** : RunPod › Settings › Secrets › nouveau secret `howe_worker_key`, valeur = la clé privée. Elle n'est écrite ni dans le dépôt, ni dans l'image.
4. **L'endpoint** : Serverless › New Endpoint › GitHub Repo, ce dépôt, branche `main`. GPU : 16 Go suffisent (Whisper small en float16). Région européenne. Workers max : 2. Idle timeout : 5 s. Execution timeout : 2 000 s.
   Variables d'environnement :
   ```
   HOWE_URL=https://howe.immo
   WORKER_REFERENCE=<référence donnée par worker-add>
   WORKER_PRIVATE_KEY={{ RUNPOD_SECRET_howe_worker_key }}
   ```
   Les autres réglages ont leur valeur dans le `Dockerfile` (compétences, places, Whisper sur GPU, arrêt après 20 s sans tâche, 30 minutes au plus).
5. **Howé** (`.env`) :
   ```
   MODERATION_RUNPOD=true
   MODERATION_RUNPOD_ENDPOINT_ID=<identifiant de l'endpoint>
   MODERATION_RUNPOD_API_KEY=<clé d'API RunPod>
   ```
   La clé d'API RunPod reste dans Howé ; le travailleur n'en a pas besoin.

**Suivre.** L'écran « Modération automatique » montre les conteneurs réveillés, leur état et leur bilan, ainsi que le travailleur « runpod » parmi les autres (on peut le vider ou le révoquer). Révoqué, un conteneur s'arrête au premier refus.

**Changer la clé.** `moderation:worker-add` d'un nouveau travailleur, mise à jour du secret et de `WORKER_REFERENCE` sur l'endpoint, puis révocation de l'ancien sur l'écran.

## Installer sur un serveur de traitement

Prérequis système, à installer avec **apt** et non avec pip : sur PyPI, « tesseract-ocr » est un autre paquet, sans rapport, qui ne se compile pas.

```bash
sudo apt install tesseract-ocr tesseract-ocr-fra libzbar0 ffmpeg
```

Les paquets Python viennent de `requirements.txt`. Un serveur qui annonce la compétence `asr` installe aussi `requirements-asr.txt` (faster-whisper, environ 300 Mo).

```bash
sudo useradd --system --home /opt/howe-moderation-worker howe-worker
sudo cp -r . /opt/howe-moderation-worker
cd /opt/howe-moderation-worker && sudo python3.12 -m venv venv && sudo venv/bin/pip install -r requirements.txt
# Serveur qui transcrit (compétence asr) :
sudo venv/bin/pip install -r requirements-asr.txt
```

Sur Howé : `php artisan moderation:worker-add vps-traitement-1 --capabilities=image`. La commande affiche une seule fois la référence et la clé privée.

```bash
sudo install -d -m 0700 -o howe-worker /etc/howe-moderation-worker
sudo install -m 0600 -o howe-worker deploy/worker.env.example /etc/howe-moderation-worker/worker.env
# Y mettre WORKER_REFERENCE ; écrire la clé privée dans /etc/howe-moderation-worker/worker.key (0600).
sudo -u howe-worker env $(sudo cat /etc/howe-moderation-worker/worker.env | xargs) venv/bin/python -m howe_worker --check
sudo cp deploy/howe-moderation-worker.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now howe-moderation-worker
```

Ensuite, sur Howé, allumer `MODERATION_MEDIA=true`.

## Retirer ou mettre à jour

- `php artisan moderation:worker drain <nom>` (ou « Vider » sur l'écran « Modération automatique ») : il ne prend plus rien et finit ses tâches ; quand l'écran ne lui en montre plus aucune en cours, `systemctl stop howe-moderation-worker`.
- `php artisan moderation:worker activate <nom>` (ou « Reprendre ») : un travailleur vidé reprend des tâches.
- `php artisan moderation:worker revoke <nom>` (ou « Révoquer ») : effet immédiat ; sa clé ne sert plus et ses tâches reviennent dans la file.
- Mise à jour : un serveur après l'autre, chacun vidé avant l'arrêt. Howé refuse les versions trop anciennes.

## Tests

```bash
python3.12 -m venv venv && venv/bin/pip install -r requirements-dev.txt
venv/bin/python -m pytest
```

Ils utilisent les vrais Tesseract, ZBar et OpenCV, qui sont légers et ne chargent aucun modèle, sur des images fabriquées. Howé, lui, est simulé.
