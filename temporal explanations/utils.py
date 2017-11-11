'''
Created on 15 Aug 2016

@author: Saumitra
'''

import numpy as np
import argparse
from sklearn.externals import joblib
import subprocess


def process_arguments_classifier_test(args):
    """ Parses the command line arguments for classifier testing.
    Args:
        args: command line arguments
    
    Return:
        dictionary object
    """
    
    parser=argparse.ArgumentParser(description='Tests/Eval a trained classifier')

    parser.add_argument('input_directory_data',
                        action='store',
                        help='path to the feature vectors and ground-truth labels')
    
    parser.add_argument('input_directory_classifier',
                        action='store',
                        help='path to the trained classifier')
    
    parser.add_argument('mean_std_file_path',
                        action='store',
                        help='path to the mean std deviation file')
    parser.add_argument('--release',
                        action='store_true',
                        default=False,
                        help='mode of execution. If given, does not need a groun-truth annotation while predicting.')

    return vars(parser.parse_args(args))


def read_testing_data(path_fv, path_gt_label, classifier_path):
    
    """ Reads the test data features, ground-truth labels and trained classifier and populates buffers
    
    Args:
        path_fv: feature vector path
        path_gt_label: gt_label path
        classifier_path: trained classifier path
    
    Return:
        ndarrays for the read feature vectors, ground-truth labels and a classifier object
    """    
    # reading feature vectors    
    with np.load(path_fv) as f:
        x_test = [f['param_%s' %i] for i in range(len(f.files))]
    
    #reading ground truth labels
    with np.load(path_gt_label) as g:
        y_test = [g['param_%s' %i] for i in range(len(g.files))]
        
    # loading the trained classifier
    clf=joblib.load(classifier_path) 

    return (x_test, y_test, clf)

def read_ffmpeg(infile, f_len, off, dur, sample_rate, cmd='/usr/local/bin/ffmpeg'):
    """
    Jan@ ISMIR 2015
    Decodes a given audio file using ffmpeg, resampled to a given sample rate,
    downmixed to mono, and converted to float32 samples. Returns a numpy array.
    """
    if f_len:
        call = [cmd, "-v", "quiet", "-i", infile, "-f", "f32le",
                "-ar", str(sample_rate), "-ac", "1", "pipe:1"]
    else:
        call = [cmd, "-v", "quiet", "-ss", str(off) ,"-i", infile, "-t", str(dur), "-f", "f32le",
                "-ar", str(sample_rate), "-ac", "1", "pipe:1"]        
    samples = subprocess.check_output(call)
    return np.frombuffer(samples, dtype=np.float32)

def metric_cal(y_true, y_pred):
    """ 
    Jan@ ISMIR 2015
    Computes classifier's performance metrics

    Args: 
        y_true: ground-truth labels
        y_pred: classfier's predictions

    Return:
        a tuple of key metric values

    """    
    preds = y_pred
    target = y_true
    nopreds = ~preds
    correct = (preds == target)
    incorrect = ~correct
    tp = (correct * preds).sum()
    fp = (incorrect * preds).sum()
    tn = (correct * nopreds).sum()
    fn = (incorrect * nopreds).sum()
    
    accuracy= float(tp+tn)/(tp+fp+tn+fn)
    precision=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    specificity = float(tn)/(tn+fp)
    f_score= (2 * precision * recall)/(precision + recall)
    
    return (accuracy, precision, recall, specificity, f_score)
  
    