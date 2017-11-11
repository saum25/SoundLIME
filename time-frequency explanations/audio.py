#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audio feature extraction routines.

Author: Jan Schlüter
"""

import sys
import os
import subprocess
import librosa
import librosa.display as disp
import matplotlib.pyplot as plt

import numpy as np
try:
    from pyfftw.builders import rfft as rfft_builder
except ImportError:
    def rfft_builder(*args, **kwargs):
        return np.fft.rfft


def read_ffmpeg(infile, sample_rate, cmd='/usr/local/bin/ffmpeg'):
    """
    Decodes a given audio file using ffmpeg, resampled to a given sample rate,
    downmixed to mono, and converted to float32 samples. Returns a numpy array.
    """
    call = [cmd, "-v", "quiet", "-i", infile, "-f", "f32le",
            "-ar", str(sample_rate), "-ac", "1", "pipe:1"]
    samples = subprocess.check_output(call)
    return np.frombuffer(samples, dtype=np.float32)

def read_ffmpeg_partial(infile, sample_rate, offset, duration, cmd='/usr/local/bin/ffmpeg'):
    """
    Decodes a given audio file using ffmpeg, resampled to a given sample rate,
    downmixed to mono, and converted to float32 samples. Returns a numpy array.
    """
    # Added support for partial file reading: -ss: offset and -t:duration
    call = [cmd, "-v", "quiet", "-ss", str(offset),"-i", infile, "-t", str(duration),"-f", "f32le",
            "-ar", str(sample_rate), "-ac", "1", "pipe:1"]
    samples = subprocess.check_output(call)
    return np.frombuffer(samples, dtype=np.float32)


def spectrogram(samples, sample_rate, frame_len, fps, batch=50):
    """
    Computes a magnitude spectrogram for a given vector of samples at a given
    sample rate (in Hz), frame length (in samples) and frame rate (in Hz).
    Allows to transform multiple frames at once for improved performance (with
    a default value of 50, more is not always better). Returns a numpy array.
    """
    if len(samples) < frame_len:
        return np.empty((0, frame_len // 2 + 1), dtype=samples.dtype)
    win = np.hanning(frame_len)
    hopsize = sample_rate // fps
    num_frames = max(0, (len(samples) - frame_len) // hopsize + 1)
    batch = min(batch, num_frames)
    if batch <= 1 or not samples.flags.c_contiguous:
        rfft = rfft_builder(samples[:frame_len], n=frame_len)
        spect = np.vstack(np.abs(rfft(samples[pos:pos + frame_len] * win))
                          for pos in range(0, len(samples) - frame_len + 1,
                                           int(hopsize)))
    else:
        rfft = rfft_builder(np.empty((batch, frame_len), samples.dtype),
                            n=frame_len, threads=1)
        frames = np.lib.stride_tricks.as_strided(
                samples, shape=(num_frames, frame_len),
                strides=(samples.strides[0] * hopsize, samples.strides[0]))
        spect = [np.abs(rfft(frames[pos:pos + batch] * win))
                 for pos in range(0, num_frames - batch + 1, batch)]
        if num_frames % batch:
            spect.extend(spectrogram(
                    samples[(num_frames // batch * batch) * hopsize:],
                    sample_rate, frame_len, fps, batch=1))
        spect = np.vstack(spect)
        
    return spect

def spectrogram_partial(samples, sample_rate, frame_len, fps, save_input, dump_path, batch=50):
    """
    Computes a magnitude spectrogram for a given vector of samples at a given
    sample rate (in Hz), frame length (in samples) and frame rate (in Hz).
    Allows to transform multiple frames at once for improved performance (with
    a default value of 50, more is not always better). Returns a numpy array.
    """
    if len(samples) < frame_len:
        return np.empty((0, frame_len // 2 + 1), dtype=samples.dtype)
    win = np.hanning(frame_len)
    hopsize = sample_rate // fps
    num_frames = max(0, (len(samples) - frame_len) // hopsize + 1)
    batch = min(batch, num_frames)
    if batch <= 1 or not samples.flags.c_contiguous:
        rfft = rfft_builder(samples[:frame_len], n=frame_len)
        spect = np.vstack((rfft(samples[pos:pos + frame_len] * win))
                          for pos in range(0, len(samples) - frame_len + 1,
                                           int(hopsize)))
    else:
        rfft = rfft_builder(np.empty((batch, frame_len), samples.dtype),
                            n=frame_len, threads=1)
        frames = np.lib.stride_tricks.as_strided(
                samples, shape=(num_frames, frame_len),
                strides=(samples.strides[0] * hopsize, samples.strides[0]))
        spect = [(rfft(frames[pos:pos + batch] * win))
                 for pos in range(0, num_frames - batch + 1, batch)]
        if num_frames % batch:
            spect.extend(spectrogram(
                    samples[(num_frames // batch * batch) * hopsize:],
                    sample_rate, frame_len, fps, batch=1))
        spect = np.vstack(spect)
        
        if save_input:
            # extract magnitude and phase from the input audio.
            # returns magnitude and phase arrays in polar form. so, spect = magnitudes * phases. to find phase just use np.exp(np.angle(D) * j * 1)
            magnitudes, phases = librosa.core.magphase(spect.T)
            '''spect_recon = magnitudes * phases  # * is element-wise multiplication
            
            # inverting            
            win_len = frame_len
            ifft_window = np.hanning(win_len)
            
            n_frames = spect_recon.shape[1]
            expected_signal_len = frame_len + hopsize * (n_frames - 1)   # How? but important
            audio_recon = np.zeros(expected_signal_len)
                
            for i in range(n_frames):
                sample = i * hopsize
                spec = spect_recon[:, i].flatten()
                spec = np.concatenate((spec.conj(), spec[-2:0:-1]), 0)  # not clear? but expands the 513 input to 1024 as DFT is symmetric
                ytmp = ifft_window * np.fft.irfft(spec, n = frame_len)
        
                audio_recon[sample:(sample + frame_len)] = audio_recon[sample:(sample + frame_len)] + ytmp
            
            librosa.output.write_wav(os.path.join(dump_path, 'input_audio_recon.wav'), audio_recon, sample_rate)'''

            # saving all the phase information to be used while reconstructing from saliency maps.
            # phases.shape: (d, t)
            np.savez(os.path.join(dump_path, 'amp'), **{'amp': magnitudes.T})
            np.savez(os.path.join(dump_path, 'phases'), **{'phases': phases.T})

        # done this as due to the previous code datatype mismatch happens while returning from function call.        
        spect = magnitudes.T
            
    # comes here two times.   
    return spect


def extract_spect(filename, sample_rate=22050, frame_len=1024, fps=70):
    """
    Extracts a magnitude spectrogram for a given audio file at a given sample
    rate (in Hz), frame length (in samples) and frame rate (in Hz). Returns a
    numpy array.
    """
    try:
        samples = read_ffmpeg(filename, sample_rate)
    except Exception:
        samples = read_ffmpeg(filename, sample_rate, cmd='avconv')
    return spectrogram(samples, sample_rate, frame_len, fps)


def extract_spect_partial(filename, save_input, dump_path, sample_rate=22050, frame_len=1024, fps=70, offset=0.0, duration=3.2):
    """
    Extracts a magnitude spectrogram for a given audio file at a given sample
    rate (in Hz), frame length (in samples) and frame rate (in Hz). Returns a
    numpy array.
    """
    try:
        samples = read_ffmpeg_partial(filename, sample_rate, offset, duration)
    except Exception:
        samples = read_ffmpeg_partial(filename, sample_rate, offset, duration, cmd='avconv')
    
    spect = spectrogram_partial(samples, sample_rate, frame_len, fps, save_input, dump_path)    

    return spect


def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq,
                          max_freq):
    """
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples.
    """
    # prepare output matrix
    input_bins = (frame_len // 2) + 1
    filterbank = np.zeros((input_bins, num_bands))

    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    spacing = (max_mel - min_mel) / (num_bands + 1)
    peaks_mel = min_mel + np.arange(num_bands + 2) * spacing
    peaks_hz = 700 * (np.exp(peaks_mel / 1127) - 1)
    fft_freqs = np.linspace(0, sample_rate / 2., input_bins)
    peaks_bin = np.searchsorted(fft_freqs, peaks_hz)

    # fill output matrix with triangular filters
    for b, filt in enumerate(filterbank.T):
        # The triangle starts at the previous filter's peak (peaks_freq[b]),
        # has its maximum at peaks_freq[b+1] and ends at peaks_freq[b+2].
        left_hz, top_hz, right_hz = peaks_hz[b:b+3]  # b, b+1, b+2
        left_bin, top_bin, right_bin = peaks_bin[b:b+3]
        # Create triangular filter compatible to yaafe
        filt[left_bin:top_bin] = ((fft_freqs[left_bin:top_bin] - left_hz) /
                                  (top_bin - left_bin))
        filt[top_bin:right_bin] = ((right_hz - fft_freqs[top_bin:right_bin]) /
                                   (right_bin - top_bin))
        filt[left_bin:right_bin] /= filt[left_bin:right_bin].sum()

    return filterbank
