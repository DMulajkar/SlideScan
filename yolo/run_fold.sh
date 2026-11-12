# usage: bash run_fold.sh <fold> [epochs]   (WSL; trains on the other runs, tests on run <fold>)
cd /mnt/c/Users/mmtes/Desktop/SlideScan
~/yolo/.venv/bin/python yolo/train_eval.py --fold $1 --epochs ${2:-25} --data $HOME/yolo/data
echo "FOLD_DONE $1"
