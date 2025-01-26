#!/bin/bash
python3 accuracy_sector_fixed_window.py 10 &
python3 accuracy_sector_fixed_window.py 20 &
python3 accuracy_sector_fixed_window.py 30 &
python3 accuracy_sector_fixed_window.py 40 &
python3 accuracy_sector_fixed_window.py 50 &
python3 accuracy_sector_fixed_window.py 60 &
python3 accuracy_sector_fixed_window.py 70 &
python3 accuracy_sector_fixed_window.py 80 &
python3 accuracy_sector_fixed_window.py 90
#python3 accuracy_sector_fixed_window.py 1000
##python3 merge_into_one_figure.py