# small helpers used by project
import os


def ensure_dir(path):
d = os.path.dirname(path)
if d:
os.makedirs(d, exist_ok=True)
