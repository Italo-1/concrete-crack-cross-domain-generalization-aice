"""Prepara los archivos de figura para Editorial Manager en paper/envio/.

Guias de AICE (JOURNAL.md): un archivo por figura, nombrados Fig1, Fig2... en
el orden en que aparecen en el texto; vectorial preferido en EPS; figuras con
fotografias (semitonos) en TIFF de al menos 300 dpi, arte combinado >= 600 dpi.

- Graficos vectoriales: EPS desde el PDF con `pdftops -eps` (MiKTeX/Poppler);
  las fuentes ya van incrustadas (05_figures.py usa pdf.fonttype 42).
- Fig. 2 y Fig. 6 contienen parches de imagen: TIFF RGB con compresion LZW
  desde el PNG a 600 dpi que escribe 05_figures.py.

El orden sale de los \\includegraphics de main.tex, no se fija a mano.

    python src/07_figuras_envio.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
PAPER = RAIZ / "paper"
FIGURAS = RAIZ / "results" / "figures"
DESTINO = PAPER / "envio"
MIKTEX = Path(r"C:\Users\ASUS\AppData\Local\Programs\MiKTeX\miktex\bin\x64")

# Figuras con fotografias: van como TIFF (semitono / arte combinado).
CON_FOTOS = {"fig1_muestras_dominios", "fig4_errores"}


def pdftops() -> str:
    encontrado = shutil.which("pdftops") or shutil.which("pdftops", path=str(MIKTEX))
    if not encontrado:
        raise SystemExit("No se encuentra pdftops (Poppler/MiKTeX).")
    return encontrado


def main() -> int:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    sin_comentarios = re.sub(r"(?<!\\)%.*", "", tex)
    rutas = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", sin_comentarios)
    DESTINO.mkdir(exist_ok=True)
    for viejo in DESTINO.glob("Fig*.*"):
        viejo.unlink()

    herramienta = pdftops()
    lineas = ["Figura | Archivo en el manuscrito | Archivo para el envío | Formato"]
    for n, ruta in enumerate(rutas, 1):
        base = Path(ruta).stem
        if base in CON_FOTOS:
            png = FIGURAS / f"{base}.png"
            salida = DESTINO / f"Fig{n}.tif"
            with Image.open(png) as im:
                dpi = im.info.get("dpi", (600, 600))
                im.convert("RGB").save(salida, compression="tiff_lzw", dpi=dpi)
            formato = f"TIFF RGB, {round(dpi[0])} dpi"
        else:
            pdf = PAPER / ruta
            salida = DESTINO / f"Fig{n}.eps"
            subprocess.run([herramienta, "-eps", str(pdf), str(salida)], check=True)
            formato = "EPS vectorial"
        lineas.append(f"Fig. {n} | {ruta} | {salida.name} | {formato}")
        print(lineas[-1])

    (DESTINO / "LEEME_figuras.txt").write_text(
        "Generado por src/07_figuras_envio.py. No editar a mano.\n\n" + "\n".join(lineas) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
