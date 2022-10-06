'''
@Modified by Paul Cho; 10th, Nov, 2020

For NTHU DSP Lab 2022 Autumn
'''
import noisereduce as nr
import pyroomacoustics as pra
from random import sample
from signal import signal
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from librosa.filters import mel as librosa_mel_fn
from scipy.fftpack import dct
from Lab1_functions_student import pre_emphasis, STFT, mel2hz, hz2mel, get_filter_banks
from Lab2_stft2audio_student import griffinlim
from scipy.fftpack import idct
from scipy.linalg import pinv

filename = './audio.wav'
source_signal, sr = sf.read(filename) #sr:sampling rate
print('Sampling rate={} Hz.'.format(sr))

### hyper parameters
# frame_length = 512                    # Frame length(samples)
# frame_step = 256                      # Step length(samples)s
frame_length = 256
frame_step = 128
emphasis_coeff = 0.95                 # pre-emphasis para
num_bands = 12        
num_bands2 = 64                # Filter number = band number
num_FFT = frame_length*2                  # FFT freq-quantization
freq_min = 0
freq_max = int(0.5 * sr)              # Nyquist
signal_length = len(source_signal)    # Signal length

# # number of frames it takes to cover the entirety of the signal
num_frames = 1 + int(np.ceil((1.0 * signal_length - frame_length) / frame_step)) #frame_length-frame_step是每次重疊的，在分的過程，通常相鄰兩個會設一定量的重疊。前面的1是要確保無論如何都有1個frame
# ##########################
# '''
# Part I:
# (1) Perform STFT on the source signal to obtain one spectrogram (with the provided STFT() function)
# (2) Pre-emphasize the source signal with pre_emphasis()
# (3) Perform STFT on the pre-emphasized signal to obtain the second spectrogram
# (4) Plot the two spectrograms together to observe the effect of pre-emphasis

# hint for plotting:
# you can use "plt.subplots()" to plot multiple figures in one.
# you can use "axis.pcolor" of matplotlib in visualizing a spectrogram. 
# '''
# #YOUR CODE STARTS HERE:
source_spectrum=STFT(source_signal,num_frames,num_FFT,frame_step,frame_length,signal_length)
pre_emphasis_signal= pre_emphasis(source_signal,emphasis_coeff)
pre_emphasis_spectrum= STFT(pre_emphasis_signal,num_frames,num_FFT,frame_step,frame_length,signal_length)
print(np.shape(pre_emphasis_spectrum))

fig, axs = plt.subplots(1,2)

axs[0].set_xlabel('frame')
axs[0].set_ylabel('frequency bands')
axs[0].pcolor(source_spectrum)
axs[1].set_xlabel('frame')
axs[1].set_ylabel('frequency bands')
axs[1].pcolor(pre_emphasis_spectrum)
plt.show()
'''
source_spectrum:是原始的signal 經過frame blocking和windowing之後再由Fast Fourier Transform轉換至頻域的表示
pre_emphasis_signal:原始signal 經過pre-emphasis後改變頻率的分亮後所產生的signal
pre_emphasis_spectrum:是pre_emphasis_signal經過frame blocking和windowing之後再由Fast Fourier Transform轉換至頻域的表示
fig, axs = plt.subplots(1,2) 要在fig上畫source_spectrum和pre_emphasis_spectrum的spectrogram
其中axs[0]是source_spectrum的spectrogram,用axs[0].pcolor(source_spectrum)畫
axs[1]是pre_emphasis_spectrum的spectrogram,用axs[1].pcolor(pre_emphasis_spectrum)畫
'''

# #YOUR CODE ENDS HERE;
# ##########################

# '''
# Head to the import source 'Lab1_functions_student.py' to complete these functions:
# mel2hz(), hz2mel(), get_filter_banks()
# '''
# # get Mel-scaled filter
# fbanks = get_filter_banks(num_bands, num_FFT , sr, freq_min, freq_max)
# ##########################
# '''
# Part II:
# (1) Convolve the pre-emphasized signal with the filter
# (2) Convert magnitude to logarithmic scale
# (3) Perform Discrete Cosine Transform (dct) as a process of information compression to obtain MFCC
#     (already implemented for you, just notice this step is here and skip to the next step)
# (4) Plot the filter banks alongside the MFCC
# '''
# #YOUR CODE STARTS HERE:
fbanks1 = get_filter_banks(num_bands, num_FFT , sr, freq_min, freq_max) # 12 banks
fbanks2 = get_filter_banks(64, num_FFT , sr, freq_min, freq_max)# 64banks
print('fbanks',np.shape(fbanks1))


features=np.dot(fbanks1,pre_emphasis_spectrum).T
features2=np.dot(fbanks2,pre_emphasis_spectrum).T 
features_log = np.log(features+1e-12)
features_log2 = np.log(features2+1e-12)
MFCC = dct(features_log, norm = 'ortho')[:,:num_bands].T #12 banks的MFCC
MFCC2 = dct(features_log2, norm = 'ortho')[:,:num_bands2].T # 64 banks的MFCC
print(np.shape(MFCC))

# step(3): Discrete Cosine Transform 
#MFCC = dct(features, norm = 'ortho')[:,:num_bands]
# equivalent to Matlab dct(x)
# The numpy array [:,:] stands for everything from the beginning to end.

fig, (ax0,ax1) = plt.subplots(1,2)
ax0.set_title('MFCC of a random variable')
ax0.set_xlabel('Cepstral conefficient')
ax0.set_ylabel('Magnitude')
ax0.plot(MFCC[:,67])
ax1.set_title('MFCC of a random variable')
ax1.set_xlabel('Cepstral conefficient')
ax1.set_ylabel('Magnitude')
ax1.plot(MFCC2[:,67])
plt.show()


fig, (ax0,ax1) = plt.subplots(1,2)
ax0.set_title('12 banks MFCC')
ax0.set_xlabel('frame')
ax0.set_ylabel('MFCC coefficient')
ax0.pcolor(MFCC)
ax1.set_title('64 banks MFCC')
ax1.set_xlabel('frame')
ax1.set_ylabel('MFCC coefficient')
ax1.pcolor(MFCC2)
plt.show()


'''
features=np.dot(fbanks,pre_emphasis_spectrum).T features是存放pre_emphasis_spectrum經過了filter banks中各個filter之後所取得的頻譜
features_log 是將feature的所有數值直接取log得到的結果
MFCC = dct(features_log, norm = 'ortho')[:,:num_bands].T 是將取log的數值經過dct後取得MFCC features
fig, (ax0,ax1) = plt.subplots(1,2) 在fig上畫  以frequency為單位的Mel-scale filter banks 和MFCC的heatmap
ax0.plot(ind/1000,i) 是用來畫出filter banks中第i個filter,ind/1000為圖中的頻率範圍
ax1.pcolor(MFCC)畫出MFCC的heatmap
plt.plot(MFCC[:,5])是畫出某一個frame的MFCC
fbanks1 表示12個banks的filter ,fbanks2表示64個banks的filter
MFCC1是由12 banks filter所取得的features, MFCC2是由64 banks filter所取得的features.
'''

############ADD THESE#################
'''
(1) Perform inverse DCT on MFCC (already done for you)
(2) Restore magnitude from logarithmic scale (i.e. use exponential)
(3) Invert the fbanks convolution
(4) Synthesize time-domain audio with Griffin-Lim
(5) Get STFT spectrogram of the reconstructed signal and compare it side by side with the original signal's STFT spectrogram
    (please convert magnitudes to logarithmic scale to better present the changes)
'''

# inverse DCT (done for you)

inv_DCT = idct(MFCC.T,n=num_bands, norm = 'ortho')
print('Shape after iDCT:', inv_DCT.shape)

# mag scale restoration:
###################
# YOUR CODE HERE
inv_features = np.exp(inv_DCT)
'''
inv_DCT的magnitude以log為單位，用e^(inv_DCT)次方轉換回來
'''
###################

# inverse convoluation against fbanks (mind the shapes of your matrices):
###################
# YOUR CODE HERE
inv_spectrogram = np.dot(np.dot(fbanks1.T,pinv(np.dot(fbanks1,fbanks1.T))),inv_features.T)#np.dot(pinv(fbanks1),features.T)

'''
藉由inverse convolution來將features轉換得到原本的STFT{signal}
'''
###################
print('Shape after inverse convolution:', inv_spectrogram.shape)


# signal restoration to time domain (You only have to finish griffinlim() in 'stft2audio_student.py'):
inv_audio = griffinlim(inv_spectrogram, n_iter=32*4, hop_length=frame_step, win_length=frame_length,)
 
#inv_audio = pra.phase.griffin_lim(inv_spectrogram.T,hop=frame_step,analysis_window=np.hamming(frame_length),n_iter=32)
sf.write('reconstructed.wav', inv_audio, samplerate=int(sr*frame_length/frame_length))
reconstructed_spectrum = STFT(inv_audio, num_frames, num_FFT, frame_step, frame_length, len(inv_audio), verbose=False)
absolute_spectrum = np.abs(source_spectrum)
# scale and plot and compare original and reconstructed signals
# scale (done for you):
absolute_spectrum = np.where(absolute_spectrum == 0, np.finfo(float).eps, absolute_spectrum)
absolute_spectrum = np.log(absolute_spectrum)
reconstructed_spectrum = np.where(reconstructed_spectrum == 0, np.finfo(float).eps, reconstructed_spectrum)
reconstructed_spectrum = np.log(reconstructed_spectrum)
'''
write the reconstrcuted audio signal to reconstructed.wav
We get the origianl signal spectrum and reconstructed signal spectrum in log scale.
'''

# fig, axs = plt.subplots(2,1)
# source_spectrum = np.log(source_spectrum+1e-12)
# axs[0].plot([i*sr/num_FFT for i in range(len(reconstructed_spectrum))],reconstructed_spectrum[:,1000])
# axs[0].set_xlabel('frequency(Hz)')
# axs[0].set_ylabel('magnitude')
# inv_audio = nr.reduce_noise(y=inv_audio, sr=sr)
# reconstructed_spectrum = np.log(STFT(inv_audio, num_frames, num_FFT, frame_step, frame_length, len(inv_audio), verbose=False)+1e-12)
# reconstructed_spectrum = np.where(reconstructed_spectrum == 0, np.finfo(float).eps, reconstructed_spectrum)
# axs[1].plot([i*sr/num_FFT for i in range(len(reconstructed_spectrum))],reconstructed_spectrum[:,1000])
# axs[1].set_xlabel('frequency(Hz)')
# axs[1].set_ylabel('magnitude')
# plt.show()


#plot:
###################
# YOUR CODE HERE
###################
fig, axs = plt.subplots(1,2)
fig.suptitle('Original signal vs. Reconstructed signal')
axs[0].set_xlabel('frame')
axs[0].set_ylabel('frequency band')
axs[0].pcolor(absolute_spectrum)
axs[1].set_xlabel('frame')
axs[1].set_ylabel('frequency band')
axs[1].pcolor(reconstructed_spectrum)
plt.show()
'''
將原始signal的頻域和重建signal的頻域的spectrogram畫出來以做比較。
'''

############ADD ABOVE#################

# #YOUR CODE ENDS HERE;
# ##########################
