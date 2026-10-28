"""Build and execute the EDA or baseline notebook from percent-format cells."""
import argparse
from pathlib import Path
import re
import sys

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from disaster_tweets.paths import PROJECT_ROOT, NOTEBOOKS_DIR, HTML_REPORTS_DIR

ROOT = PROJECT_ROOT
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('analysis', nargs='?', choices=['eda', 'baseline'], default='eda')
args = parser.parse_args()
source_name, output_stem = {'eda': ('eda.py', '01_eda'),
                            'baseline': ('baseline.py', '02_baseline')}[args.analysis]
source = (ROOT / 'scripts' / source_name).read_text(encoding='utf-8')
cells = []
for section in re.split(r'^# %%', source, flags=re.MULTILINE)[1:]:
    heading, _, body = section.partition('\n')
    if '[markdown]' in heading:
        markdown = '\n'.join(line[2:] if line.startswith('# ') else '' if line == '#' else line
                             for line in body.splitlines()).strip()
        cells.append(nbformat.v4.new_markdown_cell(markdown))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata['kernelspec'] = {'display_name': 'Python (project .venv)', 'language': 'python', 'name': 'python3'}
notebook.metadata['language_info'] = {'name': 'python', 'version': sys.version.split()[0]}
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
HTML_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
path = NOTEBOOKS_DIR / f'{output_stem}.ipynb'
nbformat.write(notebook, path)
client = NotebookClient(notebook, timeout=600, resources={'metadata': {'path': str(ROOT)}},
                        kernel_name='python3')
client.km = client.create_kernel_manager()
client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client.execute()
nbformat.write(notebook, path)
exporter = HTMLExporter()
exporter.exclude_input = False
body, _ = exporter.from_notebook_node(notebook)
(HTML_REPORTS_DIR / f'{output_stem}.html').write_text(body, encoding='utf-8')
print(f'Executed {len(cells)} cells; wrote {path.name} and {output_stem}.html')
