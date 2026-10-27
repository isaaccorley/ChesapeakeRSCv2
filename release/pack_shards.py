"""Pack each numbered patch directory into a tar archive for Hugging Face.

GeoTIFFs already use ZSTD compression, so the archives add no compression layer.
Extraction metadata directories are excluded.
"""

import argparse
import os
import tarfile
from multiprocessing import Pool


def pack_one(args: tuple) -> str:
    shard_dir, out_dir = args
    shard_name = os.path.basename(shard_dir)
    out_path = os.path.join(out_dir, f"shard-{shard_name}.tar")
    if os.path.exists(out_path):
        return f"skip (exists): {out_path}"
    tmp_path = out_path + ".tmp"
    with tarfile.open(tmp_path, "w") as tar:
        tar.add(shard_dir, arcname=shard_name)
    os.rename(tmp_path, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    return f"done: {out_path} ({size_mb:.0f} MB)"


def main(a: argparse.Namespace) -> None:
    os.makedirs(a.out_dir, exist_ok=True)
    shard_dirs = sorted(
        os.path.join(a.patches_dir, d)
        for d in os.listdir(a.patches_dir)
        if d.isdecimal() and os.path.isdir(os.path.join(a.patches_dir, d))
    )
    print(
        f"packing {len(shard_dirs)} shards from {a.patches_dir} -> {a.out_dir} "
        f"with {a.workers} workers",
        flush=True,
    )
    with Pool(a.workers) as pool:
        for result in pool.imap_unordered(
            pack_one, [(d, a.out_dir) for d in shard_dirs]
        ):
            print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patches_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--workers", type=int, default=16)
    main(p.parse_args())
