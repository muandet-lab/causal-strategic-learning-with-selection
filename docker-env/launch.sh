#!/bin/bash

# Usage: bash launch.sh ["nb"|"int"]
# Options are "nb" (notebook) or "int" (interactive)
# E.g.,
#       bash docker-env/launch.sh nb
#       bash docker-env/launch.sh int

# User's params
IMG_TAG="csl-img"
VIRTUAL_DIR="/home/guest/csl-code"
# end of user's params

set -e

# process params
docker_file=$(readlink -f -- "$(dirname "$0")""/Dockerfile")
project_dir=$(readlink -f -- "$(dirname "$0")""/..")
launch_mode=$1

# build
echo "---------------------------------"
echo "Building with context: ""$project_dir"
echo "---------------------------------"
docker build -t ${IMG_TAG} "$project_dir" -f "$docker_file"

# run
echo "---------------------------------"
echo "Binding: ${project_dir} and ${VIRTUAL_DIR}"
echo "---------------------------------"

if [[ $launch_mode = "int" ]]; then
  docker run --rm -v "$project_dir":$VIRTUAL_DIR -it ${IMG_TAG} /bin/bash
elif [[ $launch_mode = "nb" ]]; then
  docker run --rm -v "$project_dir":$VIRTUAL_DIR -p 8888:8888 ${IMG_TAG} /bin/bash -c "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"
else
  echo "Doing nothing!"
fi
