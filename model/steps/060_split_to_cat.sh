#!/bin/bash

cd wrk/
# Get the current directory
current_directory=$(pwd)

# Create a list of all of the directories in the current directory
directories=("$current_directory"/mixed/*)

# Loop through the directories
for directory in "${directories[@]}"; do

  x=$(basename $directory)

  # Check the last letter of the string
  last_letter="${x: -1}"

  echo $last_letter

  # If the last letter is not 's', add 's' to the end
  if [[ $last_letter != "s" ]]; then
    x="${x}s"
  fi

  array=(${x//-/ })
  echo "${array[0]}" "${array[1]}" "${array[2]}" "${array[3]}"

  for i in {0..3}; do
    mkdir -p _cat$i/${array[$i]};
    # Loop through the images in the directory
    for image in "$directory"/*.jpg; do
      # Create a hard link to the image in the new directory
      ln -s "$image" _cat$i/${array[$i]}/
    done
  done

  x2=$(basename $directory)
  array2=(${x2//-/ })
  #special case - everythin except color
  mkdir -p _3cat/${array2[0]}-${array2[2]}-${array2[3]};
  for image in "$directory"/*.jpg; do
      # Create a hard link to the image in the new directory
      ln -s "$image" _3cat/${array2[0]}-${array2[2]}-${array2[3]}/
  done
done
