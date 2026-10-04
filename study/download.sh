#!/bin/sh
# Downloads every source in sources.json to $RAW (<= 720p video only, like Peter's FORMAT), skipping ones present.
W="${MMA_WORK:-work}"
RAW="$W/mcgregor/raw"; mkdir -p "$RAW"
FF=$("$W/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$W/venv/bin/python" -c "import json;[print(f['id'],f['youtube']) for f in json.load(open('$(dirname "$0")/sources.json'))['fights']]" | while read id yt; do
  [ -s "$RAW/$id.mp4" ] && { echo "have $id"; continue; }
  echo "get $id $yt"
  "$W/venv/bin/yt-dlp" -q --no-progress --js-runtimes node:node --ffmpeg-location "$FF" -f "bv*[height<=720][ext=mp4]/b[height<=720][ext=mp4]/bv*[height<=720]" \
     --write-info-json -o "$RAW/$id.%(ext)s" "https://www.youtube.com/watch?v=$yt" || echo "FAILED $id"
  sleep 4
done
echo ALL-DONE
