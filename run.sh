#!/usr/bin/env bash


for filename in ./*.json
do
    target/release/naive_body25_pose_fill "${filename}"
done
