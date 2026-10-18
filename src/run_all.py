"""Ejecuta el pipeline completo con un solo comando y deja registro en results/logs/.

    python src/run_all.py              # corre todos los pasos en orden
    python src/run_all.py --from 04    # retoma desde el primer paso cuyo script empieza por 04
    python src/run_all.py --list       # muestra los pasos sin ejecutarlos
    python src/run_all.py --env-only   # solo registra versiones de librerias

Cada corrida escribe results/logs/run_<fecha>.log con la version de Python, la
version instalada de cada paquete de requirements.txt y el tiempo de cada paso.

Los entrenamientos necesitan CUBLAS_WORKSPACE_CONFIG=:4096:8 para ser
deterministas; 03_experiment.py lo fija si no esta definido.
"""
import argparse
import datetime as dt
import importlib.metadata as md
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
LOG_DIR = ROOT / "results" / "logs"
PROYECTO = "concrete-crack-cross-domain-generalization-aice"

# Orden real de ejecucion. Algunos scripts producen tablas que lee un script de
# numero menor, por eso la lista es explicita y no se ordena por nombre.
PASOS = [
    ["01_download.py"],
    ["02_preprocess.py"],
    ["02b_particion_agrupada.py"],
    ["03_experiment.py", "--condicion", "todo"],
    ["03_experiment.py", "--condicion", "escala"],
    ["03_experiment.py", "--condicion", "baseline", "--particion", "agrupada"],
    ["03b_sonda_lineal.py"],
    ["03c_multifuente.py"],
    ["tabla_arquitecturas.py"],
    ["08_umbral_auroc.py"],
    ["09_ronda2.py"],
    ["04_stats.py"],
    ["05_figures.py"],
    ["06_apendice.py"],
    ["07_figuras_envio.py"],
]


def environment_lines():
    lines = [f"python {sys.version.split()[0]} ({platform.platform()})"]
    req = ROOT / "requirements.txt"
    for raw in req.read_text(encoding="utf-8").splitlines():
        name = re.split(r"[<>=!~\[ ]", raw.strip())[0]
        if not name or name.startswith("#"):
            continue
        try:
            lines.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            lines.append(f"{name}: NO INSTALADO")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--env-only", action="store_true")
    a = ap.parse_args()

    pasos = PASOS
    if a.start:
        inicio = next((i for i, p in enumerate(PASOS) if p[0].startswith(a.start)), None)
        if inicio is None:
            sys.exit(f"Ningun paso empieza por {a.start!r}")
        pasos = PASOS[inicio:]
    if a.list:
        for p in pasos:
            print("python src/" + " ".join(p))
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / (f"environment_{stamp}.log" if a.env_only else f"run_{stamp}.log")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {PROYECTO} - {dt.datetime.now().isoformat(timespec='seconds')}\n\n## Entorno\n")
        log.write("\n".join(environment_lines()) + "\n")
        if a.env_only:
            print(f"Entorno registrado en {log_path}")
            return
        log.write("\n## Pasos\n")
        total = time.time()
        for paso in pasos:
            nombre = " ".join(paso)
            t0 = time.time()
            print(f"==> {nombre}")
            r = subprocess.run([sys.executable, str(SRC / paso[0]), *paso[1:]], cwd=ROOT)
            secs = time.time() - t0
            log.write(f"{nombre}\t{'OK' if r.returncode == 0 else 'ERROR ' + str(r.returncode)}\t{secs:.1f} s\n")
            log.flush()
            if r.returncode != 0:
                print(f"Fallo {nombre}; ver {log_path}")
                sys.exit(r.returncode)
        log.write(f"\nTotal: {time.time() - total:.1f} s\n")
    print(f"Pipeline completo. Registro: {log_path}")


if __name__ == "__main__":
    main()
