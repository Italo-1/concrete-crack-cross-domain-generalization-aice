"""Lanzador de F1 sin consola: corre los pasos en serie y reintenta con --reanudar.

Sustituye a `lanzar_f1.bat`. El 02/10 la cadena lanzada con `cmd` murio dos
veces con `^C` (una interrupcion de consola de origen no identificado) y dos
pasos fallaron por un bloqueo transitorio de archivo (README, bitacora). Aqui:

- este lanzador se arranca con `pythonw.exe` (sin consola) mediante WMI;
- cada paso corre como proceso hijo con `DETACHED_PROCESS` y su propio grupo
  de procesos: sin consola no puede recibir un Ctrl+C;
- si un paso termina con error, se reintenta (hasta 3 veces) con
  `--reanudar`, que salta los modelos ya completos;
- nunca hay dos entrenamientos a la vez en la GPU.

Arranque (PowerShell):
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = 'C:\\lineaB\\.venv\\Scripts\\pythonw.exe C:\\lineaB\\06-aice-P1\\src\\lanzar_f1.py';
        CurrentDirectory = 'C:\\lineaB\\06-aice-P1' }
Progreso: results/logs/lanzar_f1.log
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PY = Path(r"C:\lineaB\.venv\Scripts\python.exe")
LOG = RAIZ / "results" / "logs" / "lanzar_f1.log"

CADENAS = {
    # F1 (02/10): la cadena original.
    "f1": [
        ("E2 sonda lineal", ["src/03b_sonda_lineal.py", "--reanudar"], "e2_stdout.txt"),
        ("E3 multi-fuente", ["src/03c_multifuente.py", "--reanudar"], "e3_stdout.txt"),
        ("E1 tres condiciones, particion original",
         ["src/03_experiment.py", "--condicion", "todo", "--particion", "original", "--reanudar"], "e1_stdout.txt"),
    ],
    # EXP-escala (03/10, F5): protocolo en el README.
    "escala": [
        ("EXP-escala, particion original",
         ["src/03_experiment.py", "--condicion", "escala", "--particion", "original", "--reanudar"],
         "escala_stdout.txt"),
    ],
}
PASOS = CADENAS[sys.argv[1] if len(sys.argv) > 1 else "f1"]
INTENTOS = 3
BANDERAS = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP


def anotar(texto: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%d/%m/%Y %H:%M:%S')} {texto}\n")


def main() -> int:
    entorno = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONIOENCODING="utf-8")
    anotar(f"LANZADOR iniciado (pid {os.getpid()})")
    for nombre, args, salida in PASOS:
        for intento in range(1, INTENTOS + 1):
            anotar(f"INICIO {nombre} (intento {intento})")
            with open(RAIZ / "results" / "logs" / salida, "a", encoding="utf-8") as f:
                codigo = subprocess.run([str(PY), *args], cwd=RAIZ, env=entorno, stdout=f,
                                        stderr=subprocess.STDOUT, creationflags=BANDERAS).returncode
            anotar(f"FIN {nombre}, codigo {codigo}")
            if codigo == 0:
                break
            time.sleep(30)
        else:
            anotar(f"ABANDONADO {nombre} tras {INTENTOS} intentos")
    anotar("LANZADOR terminado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
