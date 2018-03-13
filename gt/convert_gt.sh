#!/bin/bash

for i in 00 01 02 03 04 05 06 07 08 09 10; do
    python convert_gt.py $i
done
