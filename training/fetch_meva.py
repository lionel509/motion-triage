"""Fetch a SMALL, capped tranche of MEVA clips (CC BY 4.0, AWS Open Data).

MEVA (mevadata.org) is 516 GB; nobody needs the firehose to start. This
pulls a few clips per camera across a few days, hard-capped by clip count
AND total bytes, renamed to the corpus filename convention with a
``meva_`` source tag so training lineage stays auditable.

::

    python training/fetch_meva.py --out data/clips/meva --max-clips 24 --max-gb 3

Attribution: "Multiview Extended Video with Activities" (MEVA) by Kitware
Inc. and IARPA, CC BY 4.0. Annotations: gitlab.kitware.com/meva/meva-data-repo.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

BUCKET = "mevadata-public-01"
NAME_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})\.(\d{2})-(\d{2})-(\d{2})\.[\d-]+\.(\w+)\.(G\d+)\.r\d+\.avi$")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--dates", nargs="*", default=["2018-03-07", "2018-03-12"])
    p.add_argument("--per-camera", type=int, default=2)
    p.add_argument("--max-clips", type=int, default=24)
    p.add_argument("--max-gb", type=float, default=3.0)
    p.add_argument("--min-mb", type=float, default=20,
                   help="skip near-empty fragments")
    p.add_argument("--max-mb", type=float, default=170)
    args = p.parse_args()

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picked, per_cam = [], defaultdict(int)
    budget = args.max_gb * 1e9
    for date in args.dates:
        token, kwargs = True, {"Bucket": BUCKET, "Prefix": f"drops-123-r13/{date}/"}
        while token:
            r = s3.list_objects_v2(**kwargs)
            for o in r.get("Contents", []):
                m = NAME_RE.search(o["Key"])
                if not m or not (args.min_mb * 1e6 <= o["Size"] <= args.max_mb * 1e6):
                    continue
                cam = m.group(6)
                if per_cam[cam] >= args.per_camera:
                    continue
                if len(picked) >= args.max_clips or \
                        sum(s for _, s, _ in picked) + o["Size"] > budget:
                    token = None
                    break
                d8 = m.group(1).replace("-", "")
                stem = (f"meva_{m.group(5)}_{cam}"
                        f"__{d8}T{m.group(2)}{m.group(3)}{m.group(4)}Z")
                picked.append((o["Key"], o["Size"], out / f"{stem}.avi"))
                per_cam[cam] += 1
            if token is None or not r.get("IsTruncated"):
                break
            kwargs["ContinuationToken"] = r["NextContinuationToken"]
        if token is None:
            break

    total = sum(s for _, s, _ in picked)
    print(f"selected {len(picked)} clips, {total/1e9:.2f} GB "
          f"({len(per_cam)} cameras) — downloading sequentially")
    for i, (key, size, dest) in enumerate(picked, 1):
        if dest.exists() and dest.stat().st_size == size:
            print(f"  [{i}/{len(picked)}] cached {dest.name}")
            continue
        s3.download_file(BUCKET, key, str(dest))
        print(f"  [{i}/{len(picked)}] {dest.name}  {size/1e6:.0f} MB")
    print(f"done -> {out}  (MEVA, CC BY 4.0 — keep the attribution line in docs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
