import os
from pathlib import Path
import logging

#setup logging
logging.basicConfig(level=logging.INFO , format='[%(asctime)s]: %(message)s:')

list_of_files=[
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.pt",
    ".env",
    "setup.py",
    "app.py",
    "research/trails.ipynb"
]

for filepath in list_of_files:
    file_path = Path(filepath)
    filedir, filename = os.path.split(file_path)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)):
        with open(filepath, "w") as f:
            pass  # create an empty file
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")