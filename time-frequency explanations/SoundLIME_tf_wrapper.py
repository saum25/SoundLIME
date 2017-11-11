'''
Created on 4 Feb 2017

@author: Saumitra

'''
"""
Some code sections in this file are take
from Jan Schluter's ismir2015 SVD code and Lasagne saliency
map recipe.
For more details check:
https://github.com/f0k/ismir2015
https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
"""

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import (MaxPool2DLayer, dropout)
import numpy as np
import matplotlib.pyplot as plt
import io
import os
floatX = theano.config.floatX
import progress
import simplecache
import audio
import model
import augment
from optparse import OptionParser
import librosa
import numpy.ma as ma
import numpy.linalg as linalg
from lime import lime_image
lime_anal = True
import time

class ModifiedBackprop(object):

    """
    Modifies the gradients flowing backward.
    The code is taken from lasagne recipe

    """

    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self.ops = {}  # memoizes an OpFromGraph instance per tensor type

    def __call__(self, x):
        # OpFromGraph is oblique to Theano optimizations, so we need to move
        # things to GPU ourselves if needed.
        if theano.sandbox.cuda.cuda_enabled:
            maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            maybe_to_gpu = lambda x: x
        # We move the input to GPU if needed.
        x = maybe_to_gpu(x)
        # We note the tensor type of the input variable to the nonlinearity
        # (mainly dimensionality and dtype); we need to create a fitting Op.
        tensor_type = x.type
        # If we did not create a suitable Op yet, this is the time to do so.
        if tensor_type not in self.ops:
            # For the graph, we create an input variable of the correct type:
            inp = tensor_type()
            # We pass it through the nonlinearity (and move to GPU if needed).
            outp = maybe_to_gpu(self.nonlinearity(inp))
            # Then we fix the forward expression...
            op = theano.OpFromGraph([inp], [outp])
            # ...and replace the gradient with our own (defined in a subclass).
            op.grad = self.grad
            # Finally, we memoize the new Op
            self.ops[tensor_type] = op
        # And apply the memoized Op to the input we got.
        return self.ops[tensor_type](x)
    
class ZeilerBackprop(ModifiedBackprop):
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        #return (grd * (grd > 0).astype(inp.dtype),)  # explicitly rectify
        return (self.nonlinearity(grd),)  # use the given nonlinearity

def opts_parser():
    usage =\
"""\r%prog: Computes predictions with a neural network trained for singing
voice detection.

Usage: %prog [OPTIONS] MODELFILE OUTFILE
  MODELFILE: file to load the learned weights from (.npz format)
  OUTFILE: file to save the prediction curves to (.npz format)  
"""
    parser = OptionParser(usage=usage)
    parser.add_option('--dataset',
            type='str', default='jamendo',
            help='Name of the dataset to use.')
    parser.add_option('--pitchshift', metavar='PERCENT',
            type='float', default=0.0,
            help='If given, perform test-time pitch-shifting of given amount '
                 'and direction in percent (e.g., -10 shifts down by 10%).')
    parser.add_option('--mem-use',
            type='choice', choices=('high', 'mid', 'low'), default='low',
            help='How much temporary memory to use. More memory allows a '
                 'faster implementation, applying the network as a fully-'
                 'convolutional net to longer excerpts or the full files.')
    parser.add_option('--cache-spectra', metavar='DIR',
            type='str', default=None,
            help='Store spectra in the given directory (disabled by default).')
    parser.add_option('--plot',
            action='store_true', default=False,
            help='If given, plot each spectrogram with predictions on screen.')

    # new command line options. added for reading and saving partial file.
    parser.add_option('--partial', 
            action = 'store_true', default=False,
            help='If given, read and predict the audio file only for the given duration and offset.')
    parser.add_option('--offset',
            type='float', default=0.0,
            help='read from the given offset location in the file for partial file read case.')
    parser.add_option('--duration',
            type='float', default=3.2,
            help='read for the given duration from the file.')
    parser.add_option('--transform', type = 'choice', default='mel', choices = ('mel', 'spect'),
                      help = 'decides the dimensions of input fed to the neural network')
    parser.add_option('--save_input',
            action='store_true', default=False,
            help='if given stores the read input audio and the extracted transform.')
    parser.add_option('--dump_path', metavar='DIR',
            type='str', default=None,
            help='Store important analysis information in given directory (disabled by default).')    
    return parser

def build_model_audio(modelfile, meanstd_file, input_dim, excerpt_size):
    
    """
    Builds the CNN architecture defined by Jan et. al. @ISMIR2015 and later loads the saved model
    and the mean std file.
    """
    # Build CNN architecture
    net = {}
    net['input'] = InputLayer((None, 1, excerpt_size, input_dim))
    kwargs = dict(nonlinearity=lasagne.nonlinearities.leaky_rectify,
                  W=lasagne.init.Orthogonal())
    net['Conv1_1'] = ConvLayer(net['input'], 64, 3, **kwargs)
    net['Conv1_2'] = ConvLayer(net['Conv1_1'], 32, 3, **kwargs)
    net['pool1'] = MaxPool2DLayer(net['Conv1_2'], 3)
    net['Conv2_1'] = ConvLayer(net['pool1'], 128, 3, **kwargs)
    net['Conv2_2'] = ConvLayer(net['Conv2_1'], 64, 3, **kwargs)
    net['pool2'] = MaxPool2DLayer(net['Conv2_2'], 3)
    net['fc3'] = DenseLayer(dropout(net['pool2'], 0.5), 256, **kwargs)
    net['fc4'] = DenseLayer(dropout(net['fc3'], 0.5), 64, **kwargs)
    net['score'] = DenseLayer(dropout(net['fc4'], 0.5), 1,
                       nonlinearity=lasagne.nonlinearities.sigmoid,
                       W=lasagne.init.Orthogonal())
    
    # load saved weights
    with np.load(modelfile) as f:
        lasagne.layers.set_all_param_values(
                net['score'], [f['param%d' % i] for i in range(len(f.files))])
        
    # - load mean/std    
    with np.load(meanstd_file) as f:
        mean = f['mean']
        std = f['std']
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)

    return net, mean, istd

def prepare_audio(mean, istd, options):

    """
    Reads input audio and creates Mel-spectrogram excerpts
    of size 115 x 80 needed by the neural network model

    """

    # default parameters from ISMIR 2015: Jan et. al.   
    sample_rate = 22050
    frame_len = 1024
    fps = 70
    mel_bands = 80
    mel_min = 27.5
    mel_max = 8000
    blocklen = 115
    
    bin_nyquist = frame_len // 2 + 1
    bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate
    
    # prepare dataset
    print("Preparing data reading...")
    datadir = os.path.join(os.path.dirname(__file__), 'dataset')

    # - load filelist
    with io.open(os.path.join(datadir, 'filelists', 'valid')) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]
    with io.open(os.path.join(datadir, 'filelists', 'test')) as f:
        filelist += [l.rstrip() for l in f if l.rstrip()]
        
    if not options.partial:
        #duration and offset arguments have not use in the part of the code.
        # - create generator for spectra
        spects = (simplecache.cached(options.cache_spectra and
                         os.path.join(options.cache_spectra, fn + '.npy'),
                         audio.extract_spect,
                         os.path.join(datadir, 'audio', fn),
                         sample_rate, frame_len, fps)
                  for fn in filelist)
    else:        
        # - create generator for spectra
        spects = (simplecache.cached(options.cache_spectra and
                         os.path.join(options.cache_spectra, fn + '.npy'),
                         audio.extract_spect_partial,
                         os.path.join(datadir, 'audio', fn),
                         options.save_input, options.dump_path, sample_rate, frame_len, fps, options.offset, options.duration)
                  for fn in filelist)

    if (options.transform == 'mel'):
        # - prepare mel filterbank
        filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                                 mel_min, mel_max)  
        
        filterbank = filterbank[:bin_mel_max].astype(floatX)
        
        # calculating and saving the pinv (80*bin_mel_max) for later use.
        filterbank_pinv = linalg.pinv(filterbank)   # pseudo inv will automatically be of shape: 80 x 372
        #filterbank_pinv = filterbank.T  # 80 x 372
        
        spects = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),
                                1e-7))
                  for spect in spects)
    
    else:
        spects = (np.log(np.maximum(spect, 1e-7))for spect in spects)
        filterbank_pinv = np.ones((mel_bands, bin_mel_max ))    # dummy of no use in this case. need to do as same code is used to return
    
    
    # - define generator for Z-scoring
    spects = ((spect - mean) * istd for spect in spects)

    # - define generator for silence-padding
    pad = np.tile((np.log(1e-7) - mean) * istd, (blocklen // 2, 1))
    spects = (np.concatenate((pad, spect, pad), axis=0) for spect in spects)
    
    # - we start the generator in a background thread (not required)
    spects = augment.generate_in_background([spects], num_cached=1)
    
    spectrum = []   # list of 3d arrays.each 3d array for one audio file No. of excerpts x 115 x 80

    # run prediction loop
    print("Generating excerpts:")
    for spect in progress.progress(spects, total=len(filelist), desc='File '):
        # - view spectrogram memory as a 3-tensor of overlapping excerpts
        num_excerpts = len(spect) - blocklen + 1
        excerpts = np.lib.stride_tricks.as_strided(
                spect, shape=(num_excerpts, blocklen, spect.shape[1]),
                strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
      
        spectrum.append(excerpts)
            
    return spectrum, filterbank_pinv

def compile_saliency_function_audio(net):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given number of input excerpts.
    """
    inp = net['input'].input_var
    outp = lasagne.layers.get_output(net['score'], deterministic=True)
    max_outp = T.max(outp, axis=1)
    saliency = theano.grad(max_outp.sum(), wrt=inp)
    #max_class = T.argmax(outp, axis=1)
    return theano.function([inp], [saliency, max_outp])

def compile_prediction_function_audio(modelfile, input_dim, excerpt_size):
    """
    Compiles a function to compute the classification prediction
    for a given number of input excerpts.
    """
    #print("Preparing prediction function...")
    # instantiate neural network
    input_var = T.tensor3('input')
    inputs = input_var.dimshuffle(0, 'x', 1, 2)  # insert "channels" dimension
    network = model.architecture(inputs, (None, 1, excerpt_size, input_dim))
     
    # load saved weights
    with np.load(modelfile) as f:
        lasagne.layers.set_all_param_values(
                network, [f['param%d' % i] for i in range(len(f.files))])

    # create output expression
    outputs = lasagne.layers.get_output(network, deterministic=True)

    # prepare and compile prediction function
    #print("Compiling prediction function...")
    return theano.function([input_var], outputs)

def gen_spect_saliency(spect, saliency_t, mean, istd, input_dim, excerpt_size, options):
    """
    reads input excerpt and its saliency, creates a masked excerpt using only the positive saliencies 
    that on a normalised scale have magnitude > 0.5.
    
    """
    
    spect_m = spect
    saliency = (saliency_t[0]).reshape(excerpt_size, input_dim)
    
    # creates a masked array based on the condition
    # here condition is magnitude of saliencies <=0 as we are interested in only positive saliencies
    # here positions where the conditions meet become 'True' in the mask. 

    sal_pos = ma.masked_where(saliency<=0, saliency)

    mask_thres = 0.5 # thresholding the saliecies to pick the most important postive saliencies.

    sal_pos.fill_value=0
    sal_data = sal_pos.filled()
    scale_sal_data = sal_data/sal_data.max()
    scale_s_d = ma.masked_where(scale_sal_data< mask_thres, scale_sal_data)
    scale_s_d_mask = scale_s_d.mask
    ss_d = spect_m * (~scale_s_d_mask)  # normalised thresholded positive saliency

    return ss_d           

# Option flags
Deconv = True

if __name__ == '__main__':
        
    # parse command line
    parser = opts_parser()
    options, args = parser.parse_args()
    if len(args) < 2:
        parser.error("missing MODELFILE and/ or MEAN_STDFILE")
    (modelfile, meanstd_file) = args
    
    #default parameters from Jan Schluter et. al. paper @ ISMIR 2015
    frame_len = 1024
    nmels = 80
    excerpt_size = 115  # equivalent to 1.6 sec
    
    # dictionary of boolean flags to guide what needs to be the output file
    # all initialised to false initially
    # 'um' - unmasked, 'm'-masked, 'cm'- conditionally masked
    flags = dict.fromkeys(['um', 'm', 'cm'], False)
    
    if (options.transform == 'mel'):
        input_dim = nmels
    else:
        input_dim = (frame_len/2)+1
        
    # Load the trained weights and reconstruct the model.  
    net, mean, istd = build_model_audio(modelfile, meanstd_file, input_dim, excerpt_size)
    
    # Generate excerpts from input audio
    # returns a "list" of 3d arrays where each element has shape (no. of excerpts) x 115 x 80
    spectrum, filterbank_pinv = prepare_audio(mean, istd, options)

    # counts number of layers with leaky_relu non-linearity
    leaky_relu = lasagne.nonlinearities.leaky_rectify
    leaky_relu_layers = [layer for layer in lasagne.layers.get_all_layers(net['score'])
       if getattr(layer, 'nonlinearity', None) is leaky_relu]
    
    if Deconv: # Deconv - Zeiler et. al. ECCV 2014
        
        modded_relu = ZeilerBackprop(leaky_relu)  # important: only instantiate this once!
        for layer in leaky_relu_layers:
            layer.nonlinearity = modded_relu
    
    else: # Default gradient - Simonyan ICLR 2013
        
        for layer in leaky_relu_layers:
            layer.nonlinearity = leaky_relu     # decides the non-linearity to be used in back-prop
        
    # compile the saliency function
    print('Compiling saliency function ....')  
    saliency_fn_audio = compile_saliency_function_audio(net)
    
    # compile the prediction function
    print('Compiling CNN prediction function ....')    
    prediction_fn_audio = compile_prediction_function_audio(modelfile, input_dim, excerpt_size)
    
    # Select random number of excerpts per audio file in test data set
    num_exc = np.random.randint(1, 100, len(spectrum))
    list_excerpts = []  # stores 2-d numpy arrays as its elements, where each element is a randomly selected excerpt of size 115 x 80   
    list_accu = []
    
    # selects the input excerpts randomly and fill a list
    for i in range(0, len(spectrum)):
        #print("Number of random excerpts selected from file %d are %d" %((i+1), num_exc[i]))
        # which excerpts are chosen per file can also be random. Here, its fixed to demonstrate for few examples
        excerpt_index_random = np.array([100]) #np.random.randint(0, spectrum[i].shape[0], num_exc[i]) 
        #print(excerpt_index_random)
        for j in excerpt_index_random:
            list_excerpts.append(spectrum[i][j])

    #print(list_excerpts[0].shape)

    # main loop: one iteration per selected excerpt
    for excerpt_id, excerpt in enumerate(list_excerpts):
        print("\n+++++++++++++++Instance index: %d+++++++++++++++++++++" %(excerpt_id+1)) 
        print ('\n--------Saliency map based analysis--------')          
        
        # extracting 2d matrix to be fed to the network saliency function
        spect = excerpt
        
        ############################SALIENCY MAP-BASED ANALYSIS#############################
        # returns a 4d saliency map (no. of excerpts 'fed' x 1 x 115 x input_dim)
        # a max_score 1d - array of shape (no. of excerpts 'fed', )
        saliency, max_score = saliency_fn_audio(spect[np.newaxis,np.newaxis,:,:])
        print('Excerpt prediction score: [%f]' %(max_score)) # near to 1 implies vocal class else non-vocal class

        # Saving the input excerpt and its saliency  
        ss_d = gen_spect_saliency(spect, saliency, mean, istd, input_dim, excerpt_size, options)
        

        ############################LIME/SLIME-BASED ANALYSIS#############################
        # We know apply SLIME to the CNN model to generate time-frequency based explanations.        
        list_exp = []

        print("\n------LIME based analysis-----")        
        explainer = lime_image.LimeImageExplainer(verbose=True)
        explanation, seg = explainer.explain_instance(image=spect, classifier_fn=prediction_fn_audio, hide_color=0, top_labels=5, num_samples=2000)
        temp, mask, fs = explanation.get_image_and_mask(0, positive_only=True, hide_rest=True, num_features=3)
        print("Top-%d components in the explanation are: (%d, %d, %d)" %(3, fs[0][0], fs[0][1], fs[0][2]))


        list_exp.append(fs) # if multiple explanations are generated for the same instance

        # saving the explanations
        plt.figure(5)
        plt.subplot(3,1,1)
        librosa.display.specshow(spect.T, y_axis= 'mel', x_axis='time', sr=22050, hop_length=315, fmin=27.5, fmax=8000)
        plt.title('mel-spectrogram[input excerpt]', fontsize=10)
        plt.subplot(3,1,2)
        librosa.display.specshow(temp.T, y_axis= 'mel', x_axis='time', sr=22050, hop_length=315, fmin=27.5, fmax=8000, cmap = 'coolwarm')
        plt.title('Top-3 interpretable components from SLIME', fontsize=10)
        plt.subplot(3,1,3)
        plt.title('Pos. saliency (grd > 0.5)', fontsize=10)
        librosa.display.specshow(ss_d.T, sr = 22050, hop_length = 315, fmin=27.5, fmax=8000, y_axis = 'mel', x_axis='time', cmap = 'OrRd')    
        plt.tight_layout()
        plt.savefig(os.path.join(options.dump_path,'analysis.eps'), format='eps', dpi=300)


    
            

        
        
        
        
        
        
        

