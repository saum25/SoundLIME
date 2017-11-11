'''
Created on 16 Aug 2016

@author: Saumitra
'''

from __future__ import print_function
import sys
import os
import numpy as np
import librosa
from lime import lime_text
import utils as U
from numpy import array
import scipy.ndimage.filters
from sklearn import metrics


# Step 1 - Given a test dataset's features and ground truth tags and a learned classifier model, classify the audio frames and evaluate the classifier performance.

# load files path
params = U.process_arguments_classifier_test(sys.argv[1:])
feature_file_path = os.path.join(params['input_directory_data'],'features_testing.npz')
ground_truth_file_path = os.path.join(params['input_directory_data'],'ground-truth_testing.npz')
classifier_path = os.path.join(params['input_directory_classifier'],'classifier.pkl')

# load mean and stddev values across each dimension    
mean_std_file = np.load(os.path.join(params['mean_std_file_path'], 'jamendo_meanstd.npz'))
mean = mean_std_file ['mean']
std = mean_std_file ['std']
#print("Mean : %f, StdDev: %f" %(mean[0], std[0]))

# load test data as a list of numpy arrays, where each member is a 2-d array (number of frames x number of feature dimensions) representing each audio file
x_testing, y_testing, clf = U.read_testing_data(feature_file_path, ground_truth_file_path, classifier_path)
x_testing_scaled = [(x-mean)*np.reciprocal(std) for x in x_testing]

print()
print("==== Multiple instance-based classifier evaluation ====")
print()

smoothen = 7    # smoothing window length
threshold = 0.55 # between vocal and non-vocal class

# generating predictions
classified_output = [ clf.predict_proba(val) for val in x_testing_scaled]
#print(classified_output[0])

# median filter-based smoothing of classifier predictions for singing voice class
predictions = [scipy.ndimage.filters.median_filter(classi[:,1], smoothen, mode='nearest') for classi in classified_output]
#print(len(predictions))
#print(predictions[0].shape)

# performance evaluation
accuracy = np.empty(len(predictions))
prec = np.empty(len(predictions))
recall = np.empty(len(predictions))
spec = np.empty(len(predictions))
f1 = np.empty(len(predictions))
accuracy_norm_val_ones=np.empty(len(predictions))

result=[]

for file_idx in range(len(predictions)):
    preds = predictions[file_idx] > threshold
    #preds = scipy.ndimage.filters.median_filter(preds, 7, mode='nearest')   # second level of smoothing: Lehner et. al. 2014 ICASSP    
    result.append(U.metric_cal(y_testing[file_idx], preds))
    accuracy_norm_val_ones [file_idx]= metrics.accuracy_score(y_testing[file_idx], np.ones(preds.shape[0], dtype=bool))

for i in range(len(result)):
    accuracy[i]=result[i][0]
    prec[i]=result[i][1]
    recall[i]=result[i][2]
    spec[i]=result[i][3]
    f1[i]=result[i][4]

print('[Jamendo test dataset] accuracy: %f prec : %f recall: %f specificity:%f f1_score: %f' % ((np.mean(accuracy)), np.mean(prec), np.mean(recall), np.mean(spec), np.mean(f1)))
print('[Jamendo test dataset] accuracy w/o learning: %f' %(np.mean(accuracy_norm_val_ones)))
    
# Step 2- Explaining the prediction of a selected instance( a.k.a frame or sample): selection can be random or manual

# Loading the audio file

print()
print('==== Instance-based classifier explanation ====')
print()
print("Input audio file [%d]: %s" %(1,'dataset/03 - Say me Good Bye.mp3'))
SR = 22050
audio_buffer = U.read_ffmpeg('./dataset/03 - Say me Good Bye.mp3', 1, 0, 30, sample_rate=SR)
print("Samples read:%d Sampling rate:%d" %(audio_buffer.size, SR)) 

# Generating temporal explanations using Sound-LIME
ss_buffer_consol = []
iterate_count = 1 # tells how many sets of temporal explanations are generated for the same instance
x_testing_file = x_testing_scaled[0] # selecting the first audio file features for further analysis.
instance_indices = [42, 179, 59, 125] # Reported in the ISMIR paper


for instance_idx in instance_indices:
    
    # classifying the selected instance        
    #reshaping is mandatory by python for 1-d arrays. Maps to a (1,no. of feature) matrix. earlier it remains (no. of features, )
    x_instance = x_testing_file[instance_idx-1].reshape(1,-1)
    prob_instance=clf.predict_proba(x_instance)
    print()
    print('[Instance id]: %d [class probabilities]: <Music> %f,<Singing Voice> %f' %(instance_idx-1, prob_instance[0][0], prob_instance[0][1]))
    if (prob_instance[0][0] > prob_instance[0][1]):
        class_pred = 0
        prob = prob_instance[0][0]
    else:
        class_pred= 1
        prob = prob_instance[0][1]

   
    # number of super-samples (temporal segments) in an instance 
    n_ss = 10
    # number of samples per instance
    n_samples = SR
    #number of samples in a super-sample
    n_samples_ss = n_samples/n_ss

    # Extract the selected instance from the audio file.
    # Reading from index number up to index-1. i.e. value at last index is not copied (e.g. for instance =1, reading happens from 0 to n_samples-1)
    seg_buffer = audio_buffer[(instance_idx-1)* n_samples: (instance_idx-1)* n_samples + n_samples]
    # save the instance
    #librosa.output.write_wav('input_audio_instance.wav', seg_buffer, SR)
    

    ss_buffer = []
    # creating a list of all temporal segmentations (ndarrays)
    for i in range(0,n_ss):
        ss_buffer.append(seg_buffer[i*n_samples_ss:n_samples_ss*(i+1)])
    
    # Using LIME/ Sound-LIME to generate temporal explanations
    class_names = ['music','singing']
    
    explainer = lime_text.LimeTextExplainer(class_names=class_names, verbose=True)
    
    exp = explainer.explain_instance(ss_buffer, clf.predict_proba, num_features=3, num_samples=1000, mean = mean, stddev=std)

    # generating explanations for 'singing voice' class
    exp_temporal_label_1=exp.as_list(label=1)
    
    print()
    print('True class: %s' % class_names[int(y_testing[0][instance_idx-1])])
    print('Predicted class:%s' % class_names[class_pred])
    print('Prediction confidence: %f' %prob)
    print()

    print('Explanation for class %s' %class_names[1])
    print('\n'.join(map(str, exp_temporal_label_1)))
    print()

