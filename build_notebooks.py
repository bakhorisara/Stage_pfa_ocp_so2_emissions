"""
Convertit les scripts `# %%` en notebooks .ipynb (structure + code + markdown).

Par securite, un .ipynb qui contient DEJA des sorties executees n'est PAS ecrase
(sinon on perdrait les figures/tableaux integres). Passer `--force` pour tout
reconstruire, puis re-executer :
    python -m jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
"""
import sys
import nbformat as nbf
from pathlib import Path

FORCE = "--force" in sys.argv

MAP = {
    "phase1_prevision.py": "01_phase1_prevision.ipynb",
    "phase2_leviers_mpc.py": "02_phase2_leviers_mpc.ipynb",
    "figures.py": "03_figures.ipynb",
}


def to_cells(src: str):
    cells, buf, is_md = [], [], False

    def flush():
        if not buf:
            return
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            return
        cells.append(nbf.v4.new_markdown_cell(text) if is_md
                     else nbf.v4.new_code_cell(text))

    for line in src.splitlines():
        if line.startswith("# %%"):
            flush()
            buf = []
            is_md = "[markdown]" in line
            continue
        buf.append(line[2:] if (is_md and line.startswith("# ")) else
                   (line[1:] if (is_md and line == "#") else line))
    flush()
    return cells


def has_outputs(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        nb = nbf.read(path, 4)
    except Exception:
        return False
    return any(c.cell_type == "code" and c.get("outputs") for c in nb.cells)


for py, ipynb in MAP.items():
    dst = Path(ipynb)
    if has_outputs(dst) and not FORCE:
        print(f"{ipynb:32s} conserve (contient des sorties ; --force pour reconstruire)")
        continue
    src = Path(py).read_text(encoding="utf-8")
    nb = nbf.v4.new_notebook()
    nb.cells = to_cells(src)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}}
    nbf.write(nb, dst)
    print(f"{ipynb:32s} <- {py}  ({len(nb.cells)} cellules)")
