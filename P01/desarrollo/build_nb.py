"""Assemble (and optionally execute) simulador_sesgos_conductuales.ipynb.

Source parts use '#%% md [tags]', '#%% code' and '#%% modules' separators.
{{key}} placeholders in markdown are filled from resultados/numeros.json when
it exists; the substituted values are written to resultados/numeros_texto.json
so the notebook can verify its own prose against the numbers it recomputes.
"""
import json, re, sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[1]  # proyecto_1; this script lives in proyecto_1/desarrollo
HERE = Path(__file__).parent
PARTS = sorted(HERE.glob("nb_part*.txt"))
MODULES = ["simlib_precios.py", "simlib_agentes.py", "simlib_estimadores.py", "simulador.py"]
MAX_LINES = 60


def segments(src: str) -> list[list[str]]:
    """Split a module at top-level lines that follow a blank line (defs, classes,
    decorators, constants, imports, comment blocks), keeping every line."""
    lines = src.rstrip("\n").split("\n")
    segs, cur = [], []
    for i, ln in enumerate(lines):
        top = ln and not ln[0].isspace() and not ln.startswith((")", "]", "}", '"""'))
        if top and i > 0 and lines[i - 1] == "" and cur:
            segs.append(cur)
            cur = []
        cur.append(ln)
    segs.append(cur)
    return segs


def module_cells(name: str) -> list[str]:
    src = (ROOT / name).read_text(encoding="utf-8")
    chunks, cur = [], []
    for seg in segments(src):
        if cur and len(cur) + len(seg) > MAX_LINES - 1:
            chunks.append(cur)
            cur = []
        cur.extend(seg)
    chunks.append(cur)
    # Each body is its chunk verbatim, blank lines included (%%writefile keeps them), so
    # concatenating the bodies reproduces the module byte for byte.
    cells = [f"%%writefile {'-a ' if i else ''}{name}\n" + "\n".join(ch) + "\n"
             for i, ch in enumerate(chunks)]
    rebuilt = "".join(c.split("\n", 1)[1] for c in cells)
    assert rebuilt == src, f"chunking changed {name}"
    return cells


def parse(text: str):
    blocks = re.split(r"^#%% ?", text, flags=re.M)
    for b in blocks:
        if not b.strip():
            continue
        header, _, body = b.partition("\n")
        yield header.strip(), body.strip("\n")


def build(nums: dict) -> nbformat.NotebookNode:
    nb = new_notebook()
    used = {}
    for part in PARTS:
        for header, body in parse(part.read_text(encoding="utf-8")):
            kind, *tags = header.split()
            if kind == "md":
                def sub(m):
                    key = m.group(1)
                    if key in nums:
                        used[key] = nums[key]
                        return str(nums[key])
                    return "…"
                body = re.sub(r"\{\{(\w+)\}\}", sub, body)
                if "preanalisis" in tags:
                    body = (ROOT / "resultados" / "pre_analisis.md").read_text(encoding="utf-8").rstrip("\n")
                cell = new_markdown_cell(body)
            elif kind == "code":
                cell = new_code_cell(body)
            elif kind == "modules":
                for m in MODULES:
                    for c in module_cells(m):
                        nb.cells.append(new_code_cell(c))
                continue
            else:
                raise ValueError(header)
            if tags:
                cell.metadata["tags"] = tags
            nb.cells.append(cell)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    (ROOT / "resultados" / "numeros_texto.json").write_text(
        json.dumps(used, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return nb


if __name__ == "__main__":
    nums_path = ROOT / "resultados" / "numeros.json"
    nums = json.loads(nums_path.read_text(encoding="utf-8")) if nums_path.exists() else {}
    nb = build(nums)
    long_cells = [(i, len(c.source.splitlines())) for i, c in enumerate(nb.cells)
                  if c.cell_type == "code" and len(c.source.splitlines()) > MAX_LINES + 1]
    print(f"{len(nb.cells)} cells; code cells over {MAX_LINES} lines: {long_cells}")
    out = ROOT / "simulador_sesgos_conductuales.ipynb"
    nbformat.write(nb, out)
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbformat.write(nb, out)
        print("executed and saved", out)
