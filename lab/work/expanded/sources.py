"""Verified official UFC YouTube sources for longitudinal fight labeling.

Metadata came from yt-dlp's public YouTube search extractor. Direct per-video
metadata/download extraction was attempted without cookies or login; YouTube
returned player errors, so this module does not claim local media availability.
"""

SOURCES = [
    {
        "video_id": "BCOy-PG8EIw",
        "title": "Conor McGregor vs Max Holloway 1 | FULL FIGHT",
        "channel": "UFC",
        "duration_seconds": 960,
        "fighters": ["Conor McGregor", "Max Holloway"],
        "fight_date": "2013-08-17",
        "event": "UFC Fight Night: Shogun vs Sonnen",
        "source_url": "https://www.youtube.com/watch?v=BCOy-PG8EIw",
        "fight_date_provenance": "https://www.ufc.com/news/notorious-cmg-ufc-fight-night-prelim-results",
        "title_claims_full_fight": True,
        "full_fight_independently_confirmed": False,
        "download_path": None,
    },
    {
        "video_id": "GoUjizSdLEw",
        "title": "Dustin Poirier vs Justin Gaethje 1 | FULL FIGHT",
        "channel": "UFC",
        "duration_seconds": 1244,
        "fighters": ["Dustin Poirier", "Justin Gaethje"],
        "fight_date": "2018-04-14",
        "event": "UFC Fight Night: Poirier vs Gaethje (Glendale)",
        "source_url": "https://www.youtube.com/watch?v=GoUjizSdLEw",
        "fight_date_provenance": "https://www.ufc.com/news/fight-night-glendale-live-results",
        "title_claims_full_fight": True,
        "full_fight_independently_confirmed": False,
        "download_path": None,
    },
    {
        "video_id": "iRpFGVPGSJc",
        "title": "Max Holloway vs Dustin Poirier 2 | FULL FIGHT | UFC Classic",
        "channel": "UFC",
        "duration_seconds": 1579,
        "fighters": ["Max Holloway", "Dustin Poirier"],
        "fight_date": "2019-04-13",
        "event": "UFC 236",
        "source_url": "https://www.youtube.com/watch?v=iRpFGVPGSJc",
        "fight_date_provenance": "https://www.ufc.com/event/ufc-236",
        "title_claims_full_fight": True,
        "full_fight_independently_confirmed": False,
        "download_path": None,
    },
    {
        "video_id": "uJBmkeUVM6s",
        "title": "Max Holloway vs Calvin Kattar | FULL FIGHT",
        "channel": "UFC Eurasia",
        "duration_seconds": 1684,
        "fighters": ["Max Holloway", "Calvin Kattar"],
        "fight_date": "2021-01-16",
        "event": "UFC Fight Night: Holloway vs Kattar (Fight Island 7)",
        "source_url": "https://www.youtube.com/watch?v=uJBmkeUVM6s",
        "fight_date_provenance": "https://www.ufc.com/news/ufc-fight-island-7-weigh-results-holloway-kattar-brown-condit",
        "title_claims_full_fight": True,
        "full_fight_independently_confirmed": False,
        "download_path": None,
    },
    {
        "video_id": "PHzT9Wl8HYQ",
        "title": "Max Holloway vs Yair Rodriguez | FULL FIGHT",
        "channel": "UFC Eurasia",
        "duration_seconds": 1777,
        "fighters": ["Max Holloway", "Yair Rodriguez"],
        "fight_date": "2021-11-13",
        "event": "UFC Fight Night: Holloway vs Rodriguez",
        "source_url": "https://www.youtube.com/watch?v=PHzT9Wl8HYQ",
        "fight_date_provenance": "https://www.ufc.com/event/ufc-fight-night-november-13-2021",
        "title_claims_full_fight": True,
        "full_fight_independently_confirmed": False,
        "download_path": None,
    },
]

METADATA_PROVENANCE = {
    "method": "yt-dlp --flat-playlist --dump-single-json ytsearch results",
    "channel_and_duration": "Fields returned by yt-dlp YouTube search entries; titles preserve the upload's own wording.",
    "public_availability": "Each exact ID appeared in live public YouTube search results returned by yt-dlp on 2026-10-01.",
    "direct_metadata_attempt": "yt-dlp --skip-download --dump-single-json on direct video URLs; YouTube returned reload/player errors. No cookies, paid source, or DRM workaround used.",
    "download_attempt": "Public yt-dlp download attempted for iRpFGVPGSJc (720p ceiling); player returned 'The page needs to be reloaded.' No file downloaded.",
    "copyright_and_training_scope": "Source discovery for user-directed fight labeling/evolution review only; no model training.",
}
