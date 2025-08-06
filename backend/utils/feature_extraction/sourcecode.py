from pathlib import Path

def load_sol_file(scr_path):
    content = ""
    if Path(scr_path).exists:
        with open(scr_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    return content
