#!/usr/bin/env bash
# One web job (called by web/server.py): upload -> frames -> COLMAP poses -> heights -> score -> replay video.
#   bash web/run_job.sh <job dir>     the job dir holds upload.<ext>; results land in <job dir>/out/
# Prints "STEP ..." lines; server.py shows the latest one on the page as progress.
# Same settings as the PC walkaround runs (Command Reference), with HDR tone-mapped first
# (an HDR video decoded straight by OpenCV reads about 0.4 mm off the PC; see SlideScan on Azure).
set -euo pipefail
J=$(realpath "$1"); R=$J/run; OUT=$J/out
P=${SLIDESCAN_PY:-$HOME/vslam/.venv/bin/python}
V=$(ls "$J"/upload.* | head -1)
cd "$(dirname "$(realpath "$0")")/.."    # SlideScan/, where board.json lives

mkdir -p "$R/raw" "$OUT"
cp board.json "$R/targets.json"           # the test board's marker layout and bolt heights
printf '{"date": "%s", "device": "web upload", "method": "walkaround", "notes": "web job"}\n' "$(date -u +%F)" > "$R/meta.json"

echo "STEP Checking the video"
TRC=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_transfer -of csv=p=0 "$V" | head -1)
TRC=${TRC%%,*}                            # iPhone files print "arib-std-b67," (an empty side-data field)
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$V" | head -1 | awk -F/ '{print $1/($2?$2:1)}')
FPS=${FPS:-30}; CLOCK=$FPS                 # frame numbers count source frames, for the replay video's clock
case "$TRC" in
  arib-std-b67|smpte2084)   # HLG (iPhone) or PQ; ponytail: PQ path untested
    echo "STEP Tone-mapping the HDR video ($TRC)"
    VF="select=not(mod(n\,3)),zscale=tin=$TRC:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
    ffmpeg -hide_banner -loglevel error -i "$V" -an -vf "$VF" -fps_mode passthrough -c:v libx264 -preset veryfast \
      -crf 12 -colorspace bt709 -color_primaries bt709 -color_trc bt709 "$R/raw/video_sdr.mp4"
    EVERY=1; CLOCK=$(awk "BEGIN{print $FPS/3}") ;;   # every 3rd frame already selected above
  *)
    ln -s "$V" "$R/raw/video.${V##*.}"
    EVERY=3 ;;
esac

echo "STEP Extracting frames and finding markers"
$P slidescan.py frames "$R" --every $EVERY
echo "STEP Solving camera poses with COLMAP"
$P methods/walkaround_sfm.py "$R" --every 3 --max-image-size 2400 --focal-35mm 26 --threads 4
echo "STEP Measuring heights"
$P methods/walkaround.py "$R"
echo "STEP Scoring against the board's known bolts"
$P score.py "$R" --method walkaround --gate --min-est 5.5 > "$OUT/score.txt"
cp "$R"/methods/walkaround/{heightmap.png,heights.json,candidates.json,scale.json} "$OUT/"
echo "STEP Rendering the reconstruction video"
# red = the same 5.5 mm cut as the score; a failed video still leaves the measurements above
$P replay.py "$R" --out "$OUT/reconstruction.mp4" --red-mm 5.5 --src-fps "$CLOCK" || { echo "video failed; results kept"; rm -f "$OUT"/reconstruction.mp4*; }
echo "STEP Done"                          # server.py then deletes the video and frames
