"""Pull a batch of recent event clips from the dashboard (DB + MinIO) for the
distillation flywheel. Runs ON the dashboard host (Mac) — needs DB + S3 env.

Writes clips as <short-uuid>.mp4 plus a manifest.csv carrying the LIVE pipeline
verdict/reason per event, so the teacher's multi-head label can later be compared
against what the live system decided (the NN<->VLM agreement harness).

Pseudonymized at the filename layer (8-char id). Raw clips are deleted after pose
extraction downstream (guardrail) — this only stages them.

Usage (on the Mac):
    set -a && source .env.native && set +a
    .venv/bin/python ~/pull_recent_clips.py 40 [outdir]
"""
from __future__ import annotations

import csv
import os
import sys

import boto3
import psycopg2
from botocore.client import Config

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.expanduser("~"), "flywheel_batch")
# mode: signal (suspicious+person first) | recent (pure recency) | door
# (door/residential-biased, to grow the starved door class).
MODE = sys.argv[3] if len(sys.argv) > 3 else "signal"
os.makedirs(OUTDIR, exist_ok=True)

ORDER = {
    "signal": "order by (llm_verdict = 'suspicious') desc, "
              "(event_subtype in ('human_body','vehicle_detected')) desc, received_at desc",
    "recent": "order by received_at desc",
    "door": "order by (task_category ilike '%door%') desc, "
            "(anomaly_reason ilike '%door%') desc, "
            "(camera_label ilike 'House%' or camera_label ilike 'Brooklyn%') desc, "
            "received_at desc",
}.get(MODE, "order by received_at desc")


def _connect():
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        try:
            return psycopg2.connect(dsn)
        except Exception as exc:  # noqa: BLE001
            print(f"DATABASE_URL connect failed ({exc}); trying localhost", flush=True)
    return psycopg2.connect(host="localhost", port=5432, dbname="dashboard",
                            user="postgres", password="postgres")


conn = _connect()
cur = conn.cursor()
cur.execute(
    f"""
    select id, camera_label, channel, event_time_utc, event_subtype,
           llm_verdict, task_category, anomaly_reason, clip_s3_key
    from events
    where clip_s3_key is not null and status = 'processed'
    {ORDER}
    limit {int(N)}
    """
)
rows = cur.fetchall()
cur.close()
conn.close()

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    config=Config(signature_version="s3v4"),
)
bucket = os.environ["S3_BUCKET"]

manifest = os.path.join(OUTDIR, "manifest.csv")
ok = 0
with open(manifest, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["short", "camera_label", "channel", "event_time_utc",
                "event_subtype", "live_verdict", "live_task_category",
                "live_anomaly_reason"])
    for (eid, cam, ch, t, sub, verdict, task, reason, key) in rows:
        short = str(eid)[:8]
        dest = os.path.join(OUTDIR, f"{short}.mp4")
        try:
            s3.download_file(bucket, key, dest)
            ok += 1
            w.writerow([short, cam, ch, t, sub, verdict, task, reason])
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {key}: {exc}", flush=True)

print(f"downloaded {ok}/{len(rows)} clips -> {OUTDIR} (+ manifest.csv)", flush=True)
