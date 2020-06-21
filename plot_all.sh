#!/bin/bash
FOLDERS=dumps/*
for folder in $FOLDERS
do
  echo "Plotting $folder/args.json"
  python3 plot.py --json_path="$folder/args.json"
done
