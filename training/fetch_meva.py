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
    p.add_argument("--max-mb", type=float, default=200)
    p.add_argument("--activity", default=None,
                   help="targeted mode: fetch only clips whose KPF annotations "
                        "contain this activity (e.g. opens_facility_door, "
                        "person_carries_heavy_object). Needs --kpf-root.")
    p.add_argument("--kpf-root", default="data/meva-data-repo/annotation")
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

    # Targeted mode: fetch ONLY clips whose KPF annotations contain --activity.
    # Far higher label yield than random dates for a specific behavior (the
    # carrying/door classes were starved until pulled this way).
    if args.activity:
        hits = set()
        for p in Path(args.kpf_root).rglob("*activities.yml"):
            try:
                if args.activity in p.read_text(encoding="utf-8", errors="ignore"):
                    hits.add(re.sub(r"[.-]activities\.yml$", "", p.name))
            except OSError:
                pass
        print(f"clips annotated with '{args.activity}': {len(hits)}")
        kpf_name = re.compile(r"(\d{4}-\d{2}-\d{2})\.(\d{2})-(\d{2})-(\d{2})\.[\d-]+\.(\w+)\.(G\d+)$")
        for stem in sorted(hits):
            if len(picked) >= args.max_clips or sum(s for _, s, _ in picked) >= budget:
                break
            km = kpf_name.match(stem)
            if not km:
                continue
            key = f"drops-123-r13/{km.group(1)}/{km.group(2)}/{stem}.r13.avi"
            try:
                size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
            except Exception:  # noqa: BLE001 — not all annotated clips are in this drop
                continue
            if not (args.min_mb * 1e6 <= size <= args.max_mb * 1e6):
                continue
            d8 = km.group(1).replace("-", "")
            tag = args.activity.replace("person_", "").split("_")[0]
            dest = out / (f"meva{tag}_{km.group(5)}_{km.group(6)}"
                          f"__{d8}T{km.group(2)}{km.group(3)}{km.group(4)}Z.avi")
            picked.append((key, size, dest))
        total = sum(s for _, s, _ in picked)
        print(f"selected {len(picked)} clips, {total/1e9:.2f} GB — downloading")
        for i, (key, size, dest) in enumerate(picked, 1):
            if dest.exists() and dest.stat().st_size == size:
                continue
            s3.download_file(BUCKET, key, str(dest))
            if i % 10 == 0 or i == len(picked):
                print(f"  [{i}/{len(picked)}] {total/1e9:.2f} GB")
        print(f"done -> {out}  (MEVA, CC BY 4.0)")
        return 0

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
