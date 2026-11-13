"""python -m howe_worker [--once] [--check]

- sans option : la boucle, jusqu'à SIGTERM (arrêt propre : finit ses tâches) ;
- --once : une prise de tâches, traitées à la suite, puis sortie ;
- --check : vérifie la configuration et la signature auprès de Howé (GET /workers/me).
"""

from __future__ import annotations

import argparse
import signal
import sys

from . import VERSION
from .client import HoweClient, HoweUnavailable, Refused
from .config import ConfigError, load
from .logs import error, log
from .runner import Runner, startup_fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="howe_worker", description="Travailleur de modération de Howé")
    parser.add_argument("--once", action="store_true", help="une prise de tâches, puis sortie")
    parser.add_argument("--check", action="store_true", help="vérifie la configuration et la signature")
    args = parser.parse_args(argv)

    try:
        settings = load()
    except ConfigError as exc:
        print(f"Configuration : {exc}", file=sys.stderr)
        error("worker.config_error", reason=str(exc))
        return 2
    client = HoweClient(settings)

    if args.check:
        try:
            me = client.me()
        except (HoweUnavailable, Refused) as exc:
            print(f"Howé : {exc}", file=sys.stderr)
            return 1
        print(f"Travailleur « {me['name']} » ({me['state']}), compétences : {', '.join(me['capabilities'])} ; version {VERSION}.")
        return 0

    runner = Runner(settings, client)
    if args.once:
        log("worker.once", taken=runner.run_once())
        return 0

    signal.signal(signal.SIGTERM, lambda *_: runner.stopping.set())
    signal.signal(signal.SIGINT, lambda *_: runner.stopping.set())
    log("worker.started", **startup_fields(settings))
    runner.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
