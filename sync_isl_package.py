import os
import shutil
import zipfile
from pathlib import Path

source_dir = Path(r"C:\Users\shubh\.gemini\antigravity\scratch\football_score_predictor")
desktop_folder = Path(r"C:\Users\shubh\OneDrive\Desktop\football_score_predictor")
desktop_zip = Path(r"C:\Users\shubh\OneDrive\Desktop\football_score_predictor.zip")
downloads_zip = Path(r"C:\Users\shubh\Downloads\football_score_predictor.zip")

print("Syncing updated files to Desktop...")
if desktop_folder.exists():
    shutil.rmtree(desktop_folder)

def ignore_patterns(path, names):
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]

shutil.copytree(source_dir, desktop_folder, ignore=ignore_patterns)

print("Updating ZIP archives...")
for z_path in [desktop_zip, downloads_zip]:
    if z_path.exists():
        z_path.unlink()
    
    with zipfile.ZipFile(z_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(desktop_folder):
            for file in files:
                full_path = Path(root) / file
                arcname = full_path.relative_to(desktop_folder)
                zipf.write(full_path, arcname)

print("Successfully packaged updated ISL project to:")
print(" - Desktop Folder:", desktop_folder)
print(" - Desktop ZIP:", desktop_zip, f"({desktop_zip.stat().st_size} bytes)")
print(" - Downloads ZIP:", downloads_zip, f"({downloads_zip.stat().st_size} bytes)")
