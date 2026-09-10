#!/bin/bash
# Downloads public datasets for towerbench. Idempotent; safe to re-run.
set -u
R=${TOWERBENCH_ROOT:-$HOME/towerbench}/data/raw
mkdir -p $R/ml-1m $R/gowalla $R/amazon $R/airbnb $R/steam
cd $R
echo "[$(date)] START downloads"
# MovieLens-1M (grouplens cert expired -> -k)
[ -f ml-1m/ml-1m.zip ] || curl -k -sS -L -o ml-1m/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
[ -f ml-1m/ml-1m/ratings.dat ] || (cd ml-1m && unzip -q -o ml-1m.zip)
echo "[$(date)] ml-1m done"
# Gowalla check-ins with lat/lon (SNAP)
[ -f gowalla/loc-gowalla_totalCheckins.txt.gz ] || curl -sS -L -o gowalla/loc-gowalla_totalCheckins.txt.gz https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
echo "[$(date)] gowalla done"
# Amazon Reviews 2023 (McAuley-Lab) 5-core ratings + item metadata (text)
HF=https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main
for c in Musical_Instruments Video_Games Software; do
  [ -f amazon/${c}.5core.csv ] || curl -sS -L -o amazon/${c}.5core.csv $HF/benchmark/5core/rating_only/${c}.csv
  [ -f amazon/meta_${c}.jsonl ] || curl -sS -L -o amazon/meta_${c}.jsonl $HF/raw/meta_categories/meta_${c}.jsonl
  echo "[$(date)] amazon $c done: $(du -sh amazon/${c}.5core.csv amazon/meta_${c}.jsonl | tr "\n" " ")"
done
# Steam (Kang & McAuley)
[ -f steam/steam_reviews.json.gz ] || curl -sS -L -o steam/steam_reviews.json.gz https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz
[ -f steam/steam_games.json.gz ] || curl -sS -L -o steam/steam_games.json.gz https://cseweb.ucsd.edu/~wckang/steam_games.json.gz
echo "[$(date)] steam done"
# Inside Airbnb: scrape current links, download listings+reviews for selected cities
curl -sS -L https://insideairbnb.com/get-the-data/ | grep -oE "https://data.insideairbnb.com/[^\"]+/(reviews|listings)\.csv\.gz" | sort -u > airbnb/all_links.txt
echo "[$(date)] airbnb links: $(wc -l < airbnb/all_links.txt)"
for city in new-york-city london paris amsterdam barcelona berlin rome los-angeles lisbon madrid; do
  for kind in listings reviews; do
    url=$(grep -E "/${city}/[0-9-]+/data/${kind}\.csv\.gz" airbnb/all_links.txt | head -1)
    [ -z "$url" ] && { echo "  no url for $city $kind"; continue; }
    out=airbnb/${city}_${kind}.csv.gz
    [ -f $out ] || curl -sS -L -o $out "$url"
    echo "  $city $kind $(du -h $out | cut -f1)  <- $url"
  done
done
echo "[$(date)] ALL DONE"
du -sh $R/*
