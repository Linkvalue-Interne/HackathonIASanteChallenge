#!/bin/bash

if [ "$#" -lt "2" ]
then
  echo "Supply input folder and output folder as arguments"
  exit 1
else
  input_folder=$1
  target_folder=$2

  mkdir -p "${target_folder}/benign"
  mkdir -p "${target_folder}/malignant"
  echo "Starting to move images to ${target_folder} \n"

  dirs=($(find "${input_folder}" -mindepth 1 -maxdepth 1 -type d -not -path '*/\.*'))
  for dir in "${dirs[@]}"; do
    cd "$dir"
    echo "\n moving ${dir}..."
    images=($(find "${PWD}" -iname "*.jpg"))
    for image in "${images[@]}"; do
      echo "$image"
      type="$(basename "$(dirname "$image")")" # either benign or malignant
      mv "$image" "$target_folder/$type"
    done
  done
  echo "\n done! \n"
fi
