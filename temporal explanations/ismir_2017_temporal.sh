echo "-----------Analysing the decision tree classifier-------------"
cp ./trained_classifier/classifer_dt_1s.pkl ./trained_classifier/classifier.pkl
python SoundLIME_temporal_wrapper.py features_groundtruth/ trained_classifier/ mean_std/
rm -rf trained_classifier/classifier.pkl
echo
echo
echo "-----------Analysing the random forest classifier-------------"
cp ./trained_classifier/classifer_rf_1s.pkl ./trained_classifier/classifier.pkl
python SoundLIME_temporal_wrapper.py features_groundtruth/ trained_classifier/ mean_std/
rm -rf trained_classifier/classifier.pkl
