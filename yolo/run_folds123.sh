# folds 1 to 3, with a 2 minute cool-down between them (laptop GPU)
for f in 1 2 3; do
  bash /mnt/c/Users/mmtes/Desktop/SlideScan/yolo/run_fold.sh $f 25 > /mnt/c/Users/mmtes/Desktop/SlideScan/yolo/fold$f.log 2>&1
  echo "fold $f exit $?"
  [ $f -lt 3 ] && sleep 120
done
echo ALL_FOLDS_DONE
