python -u 1_generate_candidate.py final
python -u 2_generate_feature.py train &
python -u 2_generate_feature.py valid &
python -u 2_generate_feature.py test &
wait
python -u 3_generate_data.py train &
python -u 3_generate_data.py valid &
python -u 3_generate_data.py test &
wait
python -u 4_deep_train.py online &
python -u 5_ranker_train.py online &
python -u boosting_train.py lgb &
python -u boosting_train.py cat &
python -u boosting_train.py xgb &

joinByChar() {
  local IFS="$1"
  shift
  echo "$*"
}
results=($(ls result/*_score.csv))
results=`joinByChar ' ' "${results[@]}"`
python -u 6_ensemble.py --files $results
