#! /bin/bash
#set -euo pipefail


main_dir="$(pwd)"
analysis_dir=/analysis
videos_dir=/videos





##python $repo_dir$src_dir$data_analysis_dir/get_mongo_videos.py
#
iter=0
for video in $(find "$main_dir$analysis_dir$videos_dir" -mindepth 1 -maxdepth 1 -type f); do
    if [[ $video == *"mp4"* ]]; then
        count=$(find "$main_dir$analysis_dir$videos_dir" -maxdepth 1 -type f|wc -l)
        iter=$(expr $iter + 1)
        echo "Video "$iter" /"$count " is started"
        echo $video;
        python $main_dir/main.py --source "$video"
        echo "================================="
    else
        echo "DS.STORE"
    fi
done

echo "ALL VIDEOS ARE ANALYSED"

