#!/bin/sh

cd ~/new/matmul-/blockTest_transpose

./plotter.py mine_2 mine_4 mine_8 mine_16 mine_32
mv timing.pdf timing_1.pdf

./plotter.py mine_64 mine_128 mine_256 mine_512 mine_1024
mv timing.pdf timing_2.pdf