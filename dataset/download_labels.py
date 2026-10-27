"""Download county label rasters and verify their sizes against the source inventory."""

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INVENTORY = Path(__file__).parent / "provenance/county_inventory.csv"


def download_county(row: dict[str, str], output_dir: Path) -> tuple[str, str]:
    """Resume an incomplete download, or skip a file of the expected size."""
    destination = output_dir / row["name"]
    expected_size = int(row["size"])
    if destination.exists() and destination.stat().st_size == expected_size:
        return row["name"], "cached"
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            row["url"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return row["name"], f"FAIL: {result.stderr.strip()[:200]}"
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        return row["name"], f"FAIL: expected {expected_size} bytes, got {actual_size}"
    return row["name"], "ok"


def main(args: argparse.Namespace) -> int:
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    years = set(args.years.split(","))
    states = set(args.states.split(",")) if args.states else set()
    with open(args.inventory) as stream:
        counties = [
            row
            for row in csv.DictReader(stream)
            if row["year"] in years and (not states or row["state"] in states)
        ]
    total_bytes = sum(int(row["size"]) for row in counties)
    print(f"{len(counties)} files, {total_bytes / 1e9:.1f} GB", flush=True)
    failures = 0
    with ThreadPoolExecutor(args.workers) as executor:
        for name, status in executor.map(
            lambda row: download_county(row, output_dir), counties
        ):
            print(name, status, flush=True)
            failures += status.startswith("FAIL")
    print(f"Finished with {failures} failed downloads.", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument(
        "--out_dir", required=True, help="Directory for county GeoTIFFs"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--years", default="2021,2022", help="Comma-separated label years"
    )
    parser.add_argument(
        "--states", default="", help="Comma-separated full state names, e.g. Maryland"
    )
    raise SystemExit(main(parser.parse_args()))
