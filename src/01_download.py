"""P1 - descarga de SDNET2018 (directa) y METU (via Kaggle).

Ver README.md, seccion "Cambio de diseño": los 4 dominios de la matriz son los
3 subconjuntos de SDNET2018 (tableros de puente D, muros W, pavimentos P) mas
METU, en vez de los 4 datasets de la ficha original (dos de los cuales resultaron
ser el mismo, y un tercero es de segmentacion, no clasificacion).

Ninguna de las dos fuentes requiere credenciales: SDNET2018 se descarga de USU
DigitalCommons y METU de Mendeley Data, su repositorio original con DOI.

    python src/01_download.py
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "data" / "raw"

SDNET_URL = (
    "https://digitalcommons.usu.edu/cgi/viewcontent.cgi"
    "?filename=2&article=1047&context=all_datasets&type=additional"
)
SDNET_TAMANO_ESPERADO_MB = 503  # ~528 284 810 bytes

# Fuente original de METU: Mendeley Data, DOI 10.17632/5y9wdsg2zt.2, CC BY 4.0.
# El identificador del archivo se obtiene de la API publica de Mendeley:
#   /public-api/datasets/5y9wdsg2zt/files?folder_id=root&version=2
METU_URL = (
    "https://data.mendeley.com/public-files/datasets/5y9wdsg2zt/files/"
    "8a70d8a5-bce9-4291-bab9-b48cfb3e87c3/file_downloaded"
)
METU_TAMANO_ESPERADO_MB = 230  # 241 363 336 bytes


def descargar_sdnet2018() -> Path:
    """Descarga y extrae SDNET2018 (~528 MB). Conserva los 3 subconjuntos D/W/P."""
    destino = DESTINO / "sdnet2018"
    marcador = destino / ".completo"

    if marcador.exists():
        print(f"  SDNET2018 ya presente en {destino}")
        return destino

    destino.mkdir(parents=True, exist_ok=True)
    zip_local = DESTINO / "SDNET2018.zip"

    if not zip_local.exists() or zip_local.stat().st_size < 500 * 1024 * 1024:
        print(f"  Descargando SDNET2018 (~{SDNET_TAMANO_ESPERADO_MB} MB)...")
        r = requests.get(SDNET_URL, timeout=900, stream=True)
        r.raise_for_status()

        total = 0
        with open(zip_local, "wb") as f:
            for trozo in r.iter_content(chunk_size=1 << 20):
                f.write(trozo)
                total += len(trozo)
                print(f"\r    {total / 1024 / 1024:.0f} MB", end="", flush=True)
        print()
    else:
        print(f"  ZIP ya descargado: {zip_local} ({zip_local.stat().st_size / 1024**2:.0f} MB)")

    print("  Extrayendo...")
    with zipfile.ZipFile(zip_local) as zf:
        zf.extractall(destino)

    # Estructura esperada tras extraer: carpetas D (deck), W (wall), P (pavement),
    # cada una con subcarpetas C (crack) / U (uncracked).
    subcarpetas = sorted(p.name for p in destino.iterdir() if p.is_dir())
    print(f"  Subconjuntos encontrados: {subcarpetas}")

    n_imagenes = sum(1 for _ in destino.rglob("*.jpg"))
    print(f"  OK: {n_imagenes} imagenes .jpg en total")

    marcador.write_text(f"ok\nsubconjuntos={subcarpetas}\nimagenes={n_imagenes}\n")
    zip_local.unlink(missing_ok=True)
    return destino


def descargar_metu() -> Path | None:
    """Descarga METU desde Mendeley Data, la fuente original.

    Se usa el repositorio original (DOI 10.17632/5y9wdsg2zt.2, CC BY 4.0) y no
    el espejo de Kaggle: es la fuente citable en el manuscrito, trae licencia
    explicita y no exige credenciales.

    El archivo viene en .rar, que Python no descomprime de fabrica. Se extrae
    con el `tar` de Windows (bsdtar sobre libarchive), que soporta lectura de
    RAR y esta disponible de serie desde Windows 10 1803.
    """
    destino = DESTINO / "metu"
    marcador = destino / ".completo"

    if marcador.exists():
        print(f"  METU ya presente en {destino}")
        return destino

    destino.mkdir(parents=True, exist_ok=True)
    rar_local = DESTINO / "metu_concrete_crack.rar"

    if not rar_local.exists() or rar_local.stat().st_size < 200 * 1024 * 1024:
        print(f"  Descargando METU (~{METU_TAMANO_ESPERADO_MB} MB) desde Mendeley Data...")
        # Se usa curl y no requests: Mendeley esta detras de Cloudflare, que
        # rechaza el cliente de requests por su huella TLS y devuelve un 403 con
        # su pagina de desafio. curl.exe viene de serie en Windows 10+ y pasa sin
        # problema. El dataset es publico y CC BY 4.0; no hay autenticacion de
        # por medio, solo un filtro de cliente.
        r = subprocess.run(
            ["curl.exe", "-L", "--fail", "--silent", "--show-error",
             "--retry", "3", "--retry-delay", "2",
             "-o", str(rar_local), METU_URL],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  FALLO la descarga: {r.stderr.strip()}")
            return None
        print(f"    {rar_local.stat().st_size / 1024 / 1024:.0f} MB descargados")
    else:
        print(f"  RAR ya descargado: {rar_local} ({rar_local.stat().st_size / 1024**2:.0f} MB)")

    print("  Extrayendo con bsdtar...")
    r = subprocess.run(
        ["tar", "-xf", str(rar_local)],
        cwd=destino, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  FALLO al extraer: {r.stderr.strip()}")
        print("  Alternativa: instalar 7-Zip y extraer manualmente en", destino)
        return None

    n_imagenes = sum(1 for _ in destino.rglob("*.jpg"))
    subcarpetas = sorted(p.name for p in destino.rglob("*") if p.is_dir())[:10]
    print(f"  OK: {n_imagenes} imagenes .jpg. Carpetas: {subcarpetas}")

    marcador.write_text(f"ok\nimagenes={n_imagenes}\nfuente=Mendeley 10.17632/5y9wdsg2zt.2\n")
    rar_local.unlink(missing_ok=True)
    return destino


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    print("=== P1: descarga de datasets de fisuras en concreto ===\n")

    print("[1/2] SDNET2018 (3 subconjuntos: tableros, muros, pavimentos)")
    try:
        descargar_sdnet2018()
    except Exception as exc:  # noqa: BLE001
        print(f"  FALLO: {type(exc).__name__}: {exc}")
        return 1

    print("\n[2/2] METU (Mendeley Data, fuente original con DOI)")
    metu_ok = descargar_metu() is not None

    if not metu_ok:
        print("\nSDNET2018 listo. METU no pudo extraerse; revisar el mensaje anterior.")
        return 2  # codigo distinto de exito total: hay trabajo pendiente, no un fallo

    print("\nDescarga completa: los 4 dominios estan disponibles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
