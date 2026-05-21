import json, re, pathlib

src = pathlib.Path("fusion_pipeline.py").read_text(encoding="utf-8")
raw_cells = re.split(r"\n(?=# ── )", src)

def make_code_cell(source_str):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [l + "\n" for l in source_str.splitlines()],
        "outputs": [],
        "execution_count": None,
    }

cells = []

# Cell 0: install
cells.append(make_code_cell(
    "# Install dependencies (run once in Colab)\n"
    "!pip install -q shap torch scikit-learn pandas matplotlib numpy\n"
    "\n"
    "# Upload data files\n"
    "# from google.colab import files\n"
    "# uploaded = files.upload()  # select: p1.npy p2.npy e1.npy e2.npy labels.npy timestamps.npy"
))

for chunk in raw_cells:
    chunk = chunk.strip()
    if chunk:
        cells.append(make_code_cell(chunk))

# Final cell: download results
cells.append(make_code_cell(
    "import zipfile, os\n"
    "from google.colab import files\n"
    "\n"
    "zip_path = 'ieee_results.zip'\n"
    "with zipfile.ZipFile(zip_path, 'w') as zf:\n"
    "    for folder in ['results', 'figures', 'models']:\n"
    "        for root, _, fnames in os.walk(folder):\n"
    "            for fname in fnames:\n"
    "                fpath = os.path.join(root, fname)\n"
    "                zf.write(fpath)\n"
    "files.download(zip_path)"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": []}
    },
    "cells": cells,
}

out = pathlib.Path("IEEE_Fusion_Pipeline.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Notebook written: {out}  ({len(cells)} cells)")
