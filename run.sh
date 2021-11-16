#!/usr/bin/env bash

rm out/*.json
cargo build --release
for filename in ./*.json
do
    target/release/naive_body25_pose_fill "${filename}"
done

cp out/*.json ../../mycompleted_data/output/keypoints3d
