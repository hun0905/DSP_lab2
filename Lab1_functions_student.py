import numpy as np
import matplotlib.pyplot as plt
def pre_emphasis(signal, coefficient = 0.95):

    return np.append(signal[0], signal[1:] - coefficient*signal[:-1])

def STFT(time_signal, num_frames, num_FFT, frame_step, frame_length, signal_length, verbose=False):
    padding_length = int((num_frames - 1) * frame_step + frame_length)
    padding_zeros = np.zeros((padding_length - signal_length,))
    padded_signal = np.concatenate((time_signal, padding_zeros))#填充要確保能分成整數個frame，且每個frame等長

    # split into frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames*frame_step, frame_step), (frame_length, 1)).T
    indices = np.array(indices,dtype=np.int32) ###找出對應到每個frame的index

    #print(np.shape(padded_signal))
    # slice signal into frames
    frames = padded_signal[indices]
    # apply window to the signal
    frames *= np.hamming(frame_length)
    # FFT
    complex_spectrum = np.fft.rfft(frames, num_FFT).T
    #print(complex_spectrum.shape)
    absolute_spectrum = np.abs(complex_spectrum)
    
    if verbose:
        print('Signal length :{} samples.'.format(signal_length))
        print('Frame length: {} samples.'.format(frame_length))
        print('Frame step  : {} samples.'.format(frame_step))
        print('Number of frames: {}.'.format(len(frames)))
        print('Shape after FFT: {}.'.format(absolute_spectrum.shape))

    return absolute_spectrum

def mel2hz(mel):
    '''
    Transfer Mel scale to Hz scale
    '''
    ###################
    # YOUR CODE HERE
    hz=(10**(mel/2595)-1)*700
    '''
    hz=(10**(mel/2595)-1)*700 就是 f = (10^(Mel/2595)-1)*700 將單位由Mel對應至hz
    '''
    ###################
    
    return hz

def hz2mel(hz):
    '''
    Transfer Hz scale to Mel scale
    '''
    ###################
    # YOUR CODE HERE
    mel=2595*np.log10(1+hz/700)
    '''
    mel=2595*np.log10(1+hz/700) 就是Mel =  2595*log_10(1+f/7000)的function 用來將單位由hz對應至Mel
    '''
    ###################

    return mel

def get_filter_banks(num_filters, num_FFT, sample_rate, freq_min = 0, freq_max = None):
    ''' Mel Bank
    num_filters: filter numbers
    num_FFT: number of FFT quantization values
    sample_rate: as the name suggests
    freq_min: the lowest frequency that mel frequency include
    freq_max: the Highest frequency that mel frequency include
    '''
    # convert from hz scale to mel scale
    low_mel = hz2mel(freq_min)
    high_mel = hz2mel(freq_max)
    # define freq-axis
    mel_freq_axis = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_freq_axis = mel2hz(mel_freq_axis)
    # Mel triangle bank design (Triangular band-pass filter banks)
    ##bin是每個頻點的頻率數，也就是sample_rate/num_FFT。代表每個FFT的點的頻率數。
    bins = np.floor((num_FFT + 1) * hz_freq_axis / sample_rate).astype(int) #=hz_freq_axis*num_FFT/sample_rate;hz_freq_axis所對到的FFT的點
    fbanks = np.zeros((num_filters, int(num_FFT / 2 + 1)))
    #fbanks = np.clip(fbanks, 1e-12, None)
    ###################
    # YOUR CODE HERE
    for m in range(num_filters):
        #fbanks[m][bins[m+1]]=1
        fbanks[m][bins[m]:bins[m+2]] = np.concatenate((np.linspace(0,1,bins[m+1]-bins[m]),np.linspace(1,0,bins[m+2]-bins[m+1])))
    '''
    總共要產生i個filter,在fbanks[i]存放第i個filter的頻域的數值
    要在第i個filter要在 bins[i]到bins[i+2]之間產生以bins[i+1]為peak的三角波
    np.linspace(0,1,bins[i+1]-bins[i])在bins[i]到bins[i+1]-1間產生由0-1的等量遞增的數列
    np.linspace(1,0,bins[i+2]-bins[i+1])在bins[i+1]到bins[i+2]-間產生由0-1的等量遞減的數列
    把np.linspace(0,1,bins[i+1]-bins[i])和np.linspace(1,0,bins[i+2]-bins[i+1])合併成一個三角波
    '''
    ###################
    return fbanks