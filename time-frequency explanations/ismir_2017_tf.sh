echo "-----------Analysing the neural network classifier-------------"
python SoundLIME_tf_wrapper.py trained_classifier/jamendo_augment_mel_Jan.npz mean_std/jamendo_meanstd.npz --offset 40 --partial --save_input --dump_path './dumps'
