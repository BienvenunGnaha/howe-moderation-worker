"""Travailleur de modération de Howé (spécification « Modération automatisée des
annonces de Howé », sections 5, 8, 9, 13 et 15 ; lots 3b et 4b).

Il ne décide jamais : il prend une tâche chez Howé, lit le fichier par un lien
signé, rend ce qu'il a lu. Howé applique ses règles et tranche.
"""

# 0.2.0 (lot 4b) : la vidéo. Howé ne confie ses tâches vidéo qu'à partir de cette version.
# 0.3.0 (lot 5) : les mesures de chaque tâche (durée, mémoire, images lues) rendues à Howé.
# 0.4.0 (lot 6) : RunPod Serverless (point d'entrée, arrêt quand il n'y a plus de tâche, Whisper sur GPU).
# 0.4.1 (13/11/2026) : journal plus détaillé (niveaux, LOG_LEVEL=debug pour chaque étape et chaque appel).
VERSION = "0.4.1"
