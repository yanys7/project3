#!/bin/bash
export PROJECT_NAME="Project3"
export PROJECT_DIR=$PWD
echo "PROJECT_DIR = $PROJECT_DIR"
echo "PROJECT_NAME = $PROJECT_NAME"

## Parallel options
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=1
