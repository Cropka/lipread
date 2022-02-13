#!/bin/bash

labels_dir=$1

for data_part in pretrain test trainval
do
  echo "Processing ${data_part} dataset"
  while IFS= read -r video_labels; do
      while IFS= read -r label; do
        mv "${label}" "${data_part}/$(basename "${video_labels}")/$(basename "${label}")"
      done <<< "$(find "${video_labels}" -mindepth 1 -maxdepth 1)"
  done <<< "$(find "${labels_dir}"/${data_part} -mindepth 1 -maxdepth 1)"
done
