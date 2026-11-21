# folds 1 to 3, with a 2 minute cool-down between them (laptop GPU)
D="$(cd "$(dirname "$0")" && pwd)"
for f in 1 2 3; do
  bash "$D/run_fold.sh" $f 25 > "$D/fold$f.log" 2>&1
  echo "fold $f exit $?"
  [ $f -lt 3 ] && sleep 120
done
echo ALL_FOLDS_DONE
