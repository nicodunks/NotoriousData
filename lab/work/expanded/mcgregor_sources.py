"""Public official McGregor fight uploads for career-stage review.

Run with work/venv/bin/python work/expanded/mcgregor_sources.py to download
the two newly selected official UFC videos (debut and Cerrone) to work/raw,
then refresh outputs/expanded/mcgregor/source_catalog_luna.json. Other entries
are metadata/provenance leads; they are not implicitly downloaded.
"""
from __future__ import annotations

import os
import hashlib
import json
import subprocess
from pathlib import Path

import cv2

ROOT = Path.cwd()
RAW = ROOT / "work/raw"
OUT = ROOT / "outputs/expanded/mcgregor/source_catalog_luna.json"
NODE = "node"

SOURCES = [
    {
        "video_id": "obgm6JNtyVo",
        "title": "Cage Warriors 51: Conor McGregor v Ivan Buchinger",
        "channel": "Cage Warriors",
        "channel_id": "UCg5E4hWlXrJX7Ukx80941FA",
        "upload_date": "2013-01-02",
        "duration_seconds": 491,
        "fight_date": "2012-12-31",
        "event": "Cage Warriors 51",
        "fighters": ["Conor McGregor", "Ivan Buchinger"],
        "source_url": "https://www.youtube.com/watch?v=obgm6JNtyVo",
        "date_provenance": "https://www.youtube.com/watch?v=obgm6JNtyVo",
        "title_claims_full_fight": False,
        "download_state": "already downloaded by parent task; do not redownload",
    },
    {
        "video_id": "HelivOF6vI8",
        "title": "Conor McGregor's DEBUT | FULL FIGHT",
        "channel": "UFC",
        "channel_id": "UCvgfXK4nTYKudb0rFR6noLA",
        "upload_date": "2020-03-25",
        "duration_seconds": 574,
        "fight_date": "2013-04-06",
        "event": "UFC on FUEL TV 9 (UFC Stockholm)",
        "fighters": ["Conor McGregor", "Marcus Brimage"],
        "source_url": "https://www.youtube.com/watch?v=HelivOF6vI8",
        "date_provenance": "https://www.ufc.com/news/mad-dog-unleashed-submits-menace-ufc-fuel-tv-9-prelim-results?language_content_entity=en",
        "title_claims_full_fight": True,
        "download_state": "selected for public <=1080p download",
    },
    {
        "video_id": "BCOy-PG8EIw",
        "title": "Conor McGregor vs Max Holloway 1 | FULL FIGHT",
        "channel": "UFC",
        "channel_id": "UCvgfXK4nTYKudb0rFR6noLA",
        "upload_date": "2023-11-22",
        "duration_seconds": 959,
        "fight_date": "2013-08-17",
        "event": "UFC Fight Night: Shogun vs Sonnen",
        "fighters": ["Conor McGregor", "Max Holloway"],
        "source_url": "https://www.youtube.com/watch?v=BCOy-PG8EIw",
        "date_provenance": "https://www.ufc.com/news/notorious-cmg-ufc-fight-night-prelim-results",
        "title_claims_full_fight": True,
        "download_state": "already downloaded by parent task; do not redownload",
    },
    {
        "video_id": "fbqepymLEe4",
        "title": "Conor McGregor vs Dustin Poirier 1",
        "channel": "ufcespanol",
        "channel_id": "UCYXJFtx4SUkrb2p_8mhLPzQ",
        "upload_date": "2024-09-27",
        "duration_seconds": 183,
        "fight_date": "2014-09-27",
        "event": "UFC 178",
        "fighters": ["Conor McGregor", "Dustin Poirier"],
        "source_url": "https://www.youtube.com/watch?v=fbqepymLEe4",
        "date_provenance": "https://www.ufc.com/athlete/conor-mcgregor?id=&page=2",
        "title_claims_full_fight": False,
        "full_fight_reference": "https://www.ufc.com/video/138565",
        "download_state": "metadata only; 183-second upload is short and needs content review",
    },
    {
        "video_id": "hoQdMji2y1M",
        "title": "Conor McGregor vs Chad Mendes | FULL FIGHT",
        "channel": "UFC",
        "channel_id": "UCvgfXK4nTYKudb0rFR6noLA",
        "upload_date": "2026-06-23",
        "duration_seconds": 849,
        "fight_date": "2015-07-11",
        "event": "UFC 189",
        "fighters": ["Conor McGregor", "Chad Mendes"],
        "source_url": "https://www.youtube.com/watch?v=hoQdMji2y1M",
        "date_provenance": "https://www.ufc.com/news/mcgregor-silences-critics-finishes-mendes-2nd",
        "title_claims_full_fight": True,
        "full_fight_reference": "https://www.ufc.com/video/141964",
        "download_state": "metadata only; candidate for later download if more footage is needed",
    },
    {
        "video_id": "7qrVQwjieac",
        "title": "Conor McGregor's KNOCKS OUT Jose Aldo | UFC 194",
        "channel": "UFC",
        "channel_id": "UCvgfXK4nTYKudb0rFR6noLA",
        "upload_date": "2021-12-12",
        "duration_seconds": 193,
        "fight_date": "2015-12-12",
        "event": "UFC 194",
        "fighters": ["Conor McGregor", "Jose Aldo"],
        "source_url": "https://www.youtube.com/watch?v=7qrVQwjieac",
        "date_provenance": "https://www.ufc.com/event/ufc-194",
        "title_claims_full_fight": False,
        "full_fight_reference": "https://www.ufc.com/video/138569",
        "download_state": "metadata only; event was decided in 13 seconds, short sample has limited longitudinal value",
    },
    {
        "video_id": "W9jPiCO35wk",
        "title": "Conor McGregor vs Nate Diaz 2 | FULL FIGHT",
        "channel": "UFC",
        "channel_id": "UCvgfXK4nTYKudb0rFR6noLA",
        "upload_date": "2026-07-09",
        "duration_seconds": 1709,
        "fight_date": "2016-08-20",
        "event": "UFC 202",
        "fighters": ["Conor McGregor", "Nate Diaz"],
        "source_url": "https://www.youtube.com/watch?v=W9jPiCO35wk",
        "date_provenance": "https://www.ufc.com/news/mcgregor-edges-diaz-five-round-epic",
        "title_claims_full_fight": True,
        "full_fight_reference": "https://www.ufc.com/video/139851",
        "download_state": "metadata only; public upload, not selected to avoid growing the raw video set excessively",
    },
    {
        "video_id": "YCE8TDYj7aU",
        "title": "Conor McGregor vs Eddie Alvarez | FULL FIGHT",
        "channel": "UFC",
        "channel_id": "UCvgfXK4nTYKudb0rFR6noLA",
        "upload_date": "2026-06-18",
        "duration_seconds": 723,
        "fight_date": "2016-11-12",
        "event": "UFC 205",
        "fighters": ["Conor McGregor", "Eddie Alvarez"],
        "source_url": "https://www.youtube.com/watch?v=YCE8TDYj7aU",
        "date_provenance": "https://www.ufc.com/news/mcgregor-wins-second-belt-alvarez-ko",
        "title_claims_full_fight": True,
        "full_fight_reference": "https://www.ufc.com/video/138571",
        "download_state": "metadata only; public upload, not selected for download",
    },
    {
        "video_id": "JuBBIJ7adjM",
        "title": "Khabib Nurmagomedov vs Conor McGregor | FULL FIGHT",
        "channel": "UFC",
        "channel_id": "UCvgfXK4nTYKudb0rFR6noLA",
        "upload_date": "2023-10-14",
        "duration_seconds": 1173,
        "fight_date": "2018-10-06",
        "event": "UFC 229",
        "fighters": ["Khabib Nurmagomedov", "Conor McGregor"],
        "source_url": "https://www.youtube.com/watch?v=JuBBIJ7adjM",
        "date_provenance": "https://www.ufc.com/athlete/conor-mcgregor",
        "title_claims_full_fight": True,
        "download_state": "already downloaded by parent task; do not redownload",
    },
    {
        "video_id": "to9GcH1mEGQ",
        "title": "Conor McGregor vs Donald Cerrone | FULL FIGHT | UFC 329",
        "channel": "UFC",
        "channel_id": "UCvgfXK4nTYKudb0rFR6noLA",
        "upload_date": "2026-06-29",
        "duration_seconds": 480,
        "fight_date": "2020-01-18",
        "event": "UFC 246",
        "fighters": ["Conor McGregor", "Donald Cerrone"],
        "source_url": "https://www.youtube.com/watch?v=to9GcH1mEGQ",
        "date_provenance": "https://www.ufc.com/event/ufc-246",
        "title_claims_full_fight": True,
        "download_state": "selected for public <=1080p download",
    },
    {
        "video_id": "6yu2AWK4rxo",
        "title": "Дастин Порье vs Конор МакГрегор 2: Вспоминаем бой",
        "channel": "UFC Eurasia",
        "channel_id": "UCU8bQExxd38i-mnn-GLOtfA",
        "upload_date": "2021-06-23",
        "duration_seconds": 859,
        "fight_date": "2021-01-23",
        "event": "UFC 257",
        "fighters": ["Dustin Poirier", "Conor McGregor"],
        "source_url": "https://www.youtube.com/watch?v=6yu2AWK4rxo",
        "date_provenance": "https://www.ufc.com/event/ufc-257",
        "title_claims_full_fight": False,
        "download_state": "already downloaded by parent task; do not redownload",
    },
]

GAPS = [
    {
        "fight": "Dustin Poirier vs Conor McGregor 3 (UFC 264, 2021-07-10)",
        "status": "No official public full-fight YouTube upload verified in this pass. UFC event/results pages confirm the bout and date; search surfaced nonofficial full-fight mirrors, which were excluded.",
        "provenance": "https://www.ufc.com/news/official-scorecards-ufc-264-poirier-vs-mcgregor-3",
    },
    {
        "fight": "Early Cage Warriors bout in 2011",
        "status": "No additional official public full-fight upload verified beyond the already downloaded 2012 Buchinger title bout. Candidate clips from unofficial channels were excluded.",
    },
]

DOWNLOAD_IDS = ["HelivOF6vI8", "to9GcH1mEGQ"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_selected() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for video_id in DOWNLOAD_IDS:
        source = next(s for s in SOURCES if s["video_id"] == video_id)
        target = RAW / f"{video_id}.mp4"
        log = ROOT / "work/expanded" / f"mcgregor-download-{video_id}.log"
        if not target.exists():
            cmd = [
                str(ROOT / "work/venv/bin/yt-dlp"),
                "--no-playlist",
                "--newline",
                "--progress-delta",
                "15",
                "--write-info-json",
                "--js-runtimes",
                f"node:{NODE}",
                "-f",
                "bestvideo[height<=1080][ext=mp4]/best[height<=1080][ext=mp4]",
                "-o",
                str(RAW / "%(id)s.%(ext)s"),
                source["source_url"],
            ]
            with log.open("w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=360, check=False)
        if not target.exists():
            source["download_result"] = {
                "path": None,
                "error_tail": log.read_text()[-1600:] if log.exists() else "download did not produce an mp4",
            }
            continue
        cap = cv2.VideoCapture(str(target))
        ok, frame = cap.read()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        source["download_result"] = {
            "path": str(target),
            "sha256": digest(target),
            "bytes": target.stat().st_size,
            "first_frame_decoded": bool(ok),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "decoded_duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else None,
            "chosen_height_limit": 1080,
        }
        cap.release()


def write_catalog() -> None:
    for source in SOURCES:
        path = RAW / f"{source['video_id']}.mp4"
        if path.exists() and "download_result" not in source:
            cap = cv2.VideoCapture(str(path))
            ok, _ = cap.read()
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            source["download_result"] = {
                "path": str(path),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
                "first_frame_decoded": bool(ok),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps,
                "decoded_duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else None,
            }
            cap.release()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    catalog = {
        "schema_version": 1,
        "created_date": "2026-10-01",
        "purpose": "Source inventory for descriptive McGregor career and movement evolution; no model training.",
        "metadata_method": "yt-dlp direct public-video metadata extraction with --skip-download --dump-single-json; no cookies or login.",
        "date_method": "Fight dates use UFC official event, results, or athlete pages; Cage Warriors 51 date is corroborated by the official Cage Warriors upload/event title and parent task's decoded source record.",
        "availability_method": "Public YouTube metadata extraction succeeded for each listed ID on 2026-10-01. UFC channel metadata is from the UFC channel ID UCvgfXK4nTYKudb0rFR6noLA; UFC Español is UCYXJFtx4SUkrb2p_8mhLPzQ.",
        "limitations": [
            "Upload title and runtime alone do not prove every bout minute is present; review decoded media before treating it as complete fight footage.",
            "The Aldo video is explicitly a knockout clip, not a complete bout; the actual fight lasted 13 seconds.",
            "The 2021 Poirier trilogy bout is date-confirmed but no official public full-fight YouTube upload was verified in this pass.",
        ],
        "sources": SOURCES,
        "coverage_gaps": GAPS,
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    tmp.replace(OUT)


if __name__ == "__main__":
    download_selected()
    write_catalog()
    print(f"Wrote {OUT}")
