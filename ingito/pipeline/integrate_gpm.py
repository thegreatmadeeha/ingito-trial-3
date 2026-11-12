import json
import shutil
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PACKS = ROOT / "ingito/packs"
GPM = ROOT / "ingito/data/gpm/processed"

REGIONS = {
    "feni": "feni-2024-08-21.pack.json",
    "south_chattogram": "south_chattogram-2026-07-12.pack.json",
}

def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prepared = []

    for region, filename in REGIONS.items():
        pack_path = PACKS / filename
        rainfall_path = GPM / f"{region}_rainfall.json"

        if not pack_path.is_file() or not rainfall_path.is_file():
            raise FileNotFoundError(f"Missing file for {region}")

        pack = json.loads(pack_path.read_text())
        rainfall = json.loads(rainfall_path.read_text())

        if not isinstance(pack, dict) or not isinstance(rainfall, dict):
            raise ValueError(f"Unexpected JSON structure for {region}")

        if "gpm_rainfall" in pack:
            raise ValueError(f"GPM already exists in {region} pack; stopping safely")

        if not rainfall.get("successful_files"):
            raise ValueError(f"No successful rainfall files for {region}")

        updated = dict(pack)
        updated["gpm_rainfall"] = {
            "dataset_id": "gpm-imerg-daily-v07-late",
            "name": "NASA GPM IMERG Daily Late Run",
            "role": "primary",
            "units": "mm/day",
            "data": rainfall,
        }

        prepared.append((region, pack_path, pack, updated))

    # Prepare all outputs before changing either pack.
    for region, path, original, updated in prepared:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n")

        check = json.loads(temp.read_text())
        check.pop("gpm_rainfall")
        if check != original:
            raise ValueError(f"Existing data changed unexpectedly in {region}")

    backup_dir = PACKS / "backups"
    backup_dir.mkdir(exist_ok=True)

    for region, path, original, updated in prepared:
        backup = backup_dir / f"{path.name}.{stamp}.bak"
        shutil.copy2(path, backup)
        os.replace(path.with_suffix(path.suffix + ".tmp"), path)
        print(f"INTEGRATED: {region}")
        print(f"BACKUP: {backup.relative_to(ROOT)}")
        print(f"Rainfall days: {updated['gpm_rainfall']['data']['successful_files']}")

    print("\nGPM INTEGRATION COMPLETE")

if __name__ == "__main__":
    main()
