"""Journal JSON sur la sortie standard (spécification, section 16) : une ligne par
événement, avec la référence de la tâche, son type, ses durées et ses mesures.

Jamais de texte lu, de contenu de fichier, de lien signé ni de clé.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any


def log(event: str, **fields: Any) -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event}
    record.update({k: v for k, v in fields.items() if v is not None})
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.stdout.flush()
