#!/bin/bash

OPTIMIZATIONS=OFF

# check provided argument
if [ "$#" -eq 1 ]; then
  ARG=$1
  if [ "$ARG" == "-o" ]; then
    OPTIMIZATIONS=On
  fi
fi

# create and enter build directory
mkdir -p build
cd build

# cmake
cmake -DOPTIMIZATIONS=$OPTIMIZATIONS ..

# build the project
make