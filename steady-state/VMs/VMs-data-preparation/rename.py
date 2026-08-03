import os
import shutil
import glob
from pathlib import Path

classification_dirpath = './data/classification'
jmh_dirpath = './data/timeseries/all'

prepared_dirpath = './data-prepared'

def rename():
    try:
        timeseries_dir = Path(prepared_dirpath) / 'timeseries'
        classification_dir = Path(prepared_dirpath) / 'classification'
        timeseries_dir.mkdir(parents=True, exist_ok=True)
        classification_dir.mkdir(parents=True, exist_ok=True)

        for timeseries_path in glob.glob(f'{jmh_dirpath}/*.json'):
            print(f"Processing: {timeseries_path}")
            filename = os.path.basename(timeseries_path)
            classification_path = os.path.join(classification_dirpath, filename)

            # ignore PyPy VM
            if "PyPy" in filename:
                print(f"Skipping PyPy virtual machine (too few data): {filename}")
                continue

            # Parse filename components
            parts = filename.split("___")
            if len(parts) != 2 or '-' not in parts[1]:
                print(f"Skipping invalid filename format: {filename}")
                continue

            hardware = parts[0]
            benchmark, vm = parts[1].split("-", 1)
            new_filename = f"{hardware}___{vm}#{benchmark}.json"

            # Construct full destination paths
            timeseries_dest = timeseries_dir / new_filename
            classification_dest = classification_dir / new_filename

            # Copy and rename files
            shutil.copy2(timeseries_path, timeseries_dest)
            shutil.copy2(classification_path, classification_dest)

    except Exception as e:
        print(f"Error during rename operation: {e}")

if __name__ == '__main__':
    rename()