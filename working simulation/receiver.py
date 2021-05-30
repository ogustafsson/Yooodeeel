import numpy as np
import csv
import pandas as pd
from scipy.linalg import dft
from scipy import fft
from scipy import ifft
import matplotlib.pyplot as plt
import bitarray
import pdb
from scipy.io.wavfile import write # For writing to a .wav file
from scipy.io.wavfile import read # For reading from a .wav file to an np array.
from scipy import signal # Module for generating chirp.

import ldpc
from bitarray import bitarray
from PIL import Image
import binascii

import transmitter

# Define constants
N = 2048 #OFDM symbol length
L = 256 #Cyclic prefix length
n = 10 # Number of repetitions of known OFDM symbol for transmission
sample_rate = 44100 # Sample rate is 44.1 kHz
FILE_NAME = 'output.wav'
# Define the QPSK constellation and its power
A = 10 # Radius of the QPSK constellation circle
seed = 2021 # Standardise random number generator seed as 2021
rng = np.random.default_rng(seed) # Random number generator with specified seed
chirp_constants = {"startFrequency": 100, "endFrequency": 20000, "duration": 0.1, "method": "linear"}
data_length = 18432 #length of data sequence in time domain
#Defining constants and mapping/demapping tables
mapping_table = {
    (0,0) : A*(1+1j),
    (0,1) : A*(-1+1j),
    (1,1) : A*(-1-1j),
    (1,0): A*(1-1j)
}
demapping_table = {v : k for k, v in mapping_table.items()}
noise_variance = 1


# Useful helper functions for converting between np arrays and wav files.
def convert_array_to_wav(array, sample_rate, FILE_NAME):
    write(FILE_NAME, sample_rate, array)

def convert_wav_to_array(wav_filename):
    rate, array = read(wav_filename)

    return array

# Functions performing the DFT and iDFT.
def DFT(signal):
    """Compute the discrete Fourier transform of a signal"""
    return fft(signal)

def iDFT(signal):
  return ifft(signal)

# Returns on the real part of the iDFT.
def iDFT_real(signal):
  inverse_transform = ifft(signal)
  return inverse_transform.real

# Functions for performing chirp synchronization.

# Generate chirp signal
def generateChirp(chirp_constants, sample_rate):
    """Generates a chirp signal with dictionary of chirp constants including: start and end frequency, duration in seconds and the method of frequency sweep"""
    number_of_samples = int(sample_rate*chirp_constants["duration"]) # Calculate the number of samples of the chirp by multiplying the duration in seconds by the sampling rate
    t_samples = np.array(range(0, number_of_samples)) # Create an array with equally spaced sample indices
    t_seconds = t_samples / sample_rate # Divide by the sampling frequency to generate the time indices in seconds

    chirp_signal = signal.chirp(t_seconds, chirp_constants["startFrequency"], chirp_constants["duration"], chirp_constants["endFrequency"], method=chirp_constants["method"]) # Use scipy library to generate chirp signal

    return chirp_signal

# Matched filter returns index of highest correlation between a noisy signal and the desired signal.
# For the chirp, this index will be located halfway along the chirp.
def matched_filter(noisy_signal, desired_signal):
    desired_signal = desired_signal.flatten()
    correlation = signal.correlate(noisy_signal, desired_signal, mode='same')
    max_correlation = np.amax(correlation)
    max_correlation_index = np.argmax(correlation)

    return max_correlation_index

# Find index of the highest correlation of a chirp signal with itself
def find_chirp_start_index(chirp_signal):
    """Uses a matched filter to find index of the highest correlation of a chirp signal with itself"""
    chirp_length = len(chirp_signal)
    chirp_start_index = matched_filter(chirp_signal, chirp_signal)
    assert chirp_start_index == int(chirp_length/2), "Chirp start index does not coincide with half of chirp length"
    return chirp_start_index

# Remove cyclic prefix
def removeCP(signal):
    """Remove the cyclic prefix of a signal"""
    return signal[L:(L+N)]

# Functions for generating the known OFDM symbol.

# Need to fix this, block size gives error with large integer
def generate_random_indices(block_size):
    """Generates numpy array of random indices with seed '2021' such that random array is predictable"""
    seed = 2021 # Standardise random number generator seed as 2021, which I arbitrarily selected
    rng = np.random.default_rng(seed) # Random number generator with specified seed
    random_constellation_indices = rng.integers(low=0, high=4, size=block_size) # Generate 511 random indices for the QPSK constellation

    return random_constellation_indices

# Generate an array of 511 random constellation symbols with '0' as the first element.
# 'A' is the radius of the QPSK constellation circle.
def generate_random_sequence(N, QPSK=True, A=10):
    """Takes the size of OFDM symbol length, assuming QPSK encoding and returns array of random constellation symbols"""
    block_size = int(N / 2 - 1) # Set size of block to OFDM symbol length divided by 2 for mirroring and subtract 1 to have zeroed first entry
    mapping = A/np.sqrt(2) * np.array([(1+1j), (-1+1j), (-1-1j), (1-1j)]) # Define the mapping of indices to QPSK constellation

    random_constellation_indices = generate_random_indices(block_size)
    random_sequence = np.array([0]) # Set up first element to be constellation symbol corresponding to zero
    for index in random_constellation_indices:
        random_sequence = np.append(random_sequence, mapping[index])

    return random_sequence

# Function for averaging two-dimensional arrays, element-wise.
def compute_average(two_dimensional_array, N, n):
    """
    Computes the average of a two-dimensional array, element-wise.
    Needs the length of the 'block' N and the number of blocks n.
    For example, inputting np.array([[0,1,2],[2,3,4]]) would return np.array([1,2,3]).
    Note, we call each individual array a 'block' so we are just averaging over them.
    (There should be a nicer 'numpyier' 'vectorised' way of doing this better, try later)
    """
    sum_of_blocks = np.zeros(N) # Initialise as zeros array for summing each block of two-dimensional array
    for block in two_dimensional_array:
        sum_of_blocks = sum_of_blocks + block # Compute sum of each block all together
    average_block = sum_of_blocks / n # Take average by dividing by number of blocks
    return average_block

# Functions for converting frequency domain channel response to time domain impulse response.
# Takes an array and appends the flipped/reversed complex-conjugate array. The two concatenated arrays both begin with a '0'.
def generate_symmetric_sequence(sequence):
    """Takes a array sequence and appends a reversed form on the end"""
    reversed_sequence = np.conj(np.append(np.array([0]), np.flip(sequence[1:int(N/2)]))) # Reverse the order of the input array, and take the complex conjugate.
    symmetric_sequence = np.append(sequence, reversed_sequence) # Append the flipped array to the original array.
    return symmetric_sequence

# Returns H_est and h_est.
def estimate_channel(known_OFDM_symbol_block_start_location, known_OFDM_symbol_block_end_location, received_signal, N, L, n):
  # Generate the known OFDM symbol.
  known_OFDM_symbol = generate_random_sequence(N)[1:]

  # Isolate the received known OFDM symbol block in the time domain signal.
  received_known_OFDM_symbol_block_time_domain = received_signal[known_OFDM_symbol_block_start_location:known_OFDM_symbol_block_end_location]

  # Loop through the received known OFDM symbols to get an estimate of the channel frequency response for each symbol.
  # Split into arrays containing each OFDM symbol in the time domain in its own array.
  array_of_individual_known_OFDM_symbols_time_domain = np.split(received_known_OFDM_symbol_block_time_domain, n)

  # Remove the cyclic prefixes.
  array_of_individual_known_OFDM_symbols_time_domain_noCP = [removeCP(OFDM_symbol_time_domain_with_CP) for OFDM_symbol_time_domain_with_CP in array_of_individual_known_OFDM_symbols_time_domain]

  # Take the DFT of each known OFDM symbol in the time domain. i.e. demodulate.
  array_of_individual_known_OFDM_symbols_frequency_domain = [DFT(OFDM_symbol_time_domain) for OFDM_symbol_time_domain in array_of_individual_known_OFDM_symbols_time_domain_noCP]

  # Keep only the relevant frequency bins. i.e. 1 to 1023.
  array_of_individual_known_OFDM_symbols_frequency_domain_relevant_half = [OFDM_symbol_frequency_domain[1:int(N/2)] for OFDM_symbol_frequency_domain in array_of_individual_known_OFDM_symbols_frequency_domain]

  # Divide by the known transmitted symbol to obtain the frequency reponse.
  array_of_frequency_response = [OFDM_demod / known_OFDM_symbol for OFDM_demod in array_of_individual_known_OFDM_symbols_frequency_domain_relevant_half]

  # Compute the averaged estimated frequency response.
  H_est = compute_average(array_of_frequency_response, int(N/2 - 1), n)

  # To get the time domain impulse response (to be real), we need to have conjugate symmetry in the frequency domain. With zeros in the right places.
  reversed_H_est = np.conj(np.append(np.array([0]), np.flip(H_est)))
  H_est_symmetric = np.append(H_est, reversed_H_est)
  H_est_symmetric = np.append(np.array([0]), H_est_symmetric)

  # Compute the impulse response.
  h_est = iDFT_real(H_est_symmetric)

  return H_est, h_est

# Returns the estimated received constellation symbols.
def demodulate(OFDM_data_symbol_block_start_location, number_of_OFDM_data_blocks, received_signal, N, L):
  estimated_received_constellation_symbols = np.array([])
  current_symbol_start_location = OFDM_data_symbol_block_start_location

  # Perform demodulation on each symbol
  for i in range(0, number_of_OFDM_data_blocks):
    # Consider the current symbol.
    current_symbol_time_domain = received_signal[current_symbol_start_location:current_symbol_start_location+L+N]

    # Remove cyclic prefix.
    current_symbol_time_domain_noCP = removeCP(current_symbol_time_domain)

    # Take the DFT.
    current_symbol_frequency_domain = DFT(current_symbol_time_domain_noCP)

    # Consider the non-repeated part of the current symbol. i.e. indices 1 to 1023.
    current_symbol_frequency_domain_relevant_part = current_symbol_frequency_domain[1:int(N/2)]

    # Obtain the estimate of the transmitted constellation symbols by dividing by the frequency response.
    estimated_received_symbol = current_symbol_frequency_domain_relevant_part / H_est

    # Append estimated received symbol to array of all received constellation symbols.
    estimated_received_constellation_symbols = np.append(estimated_received_constellation_symbols, estimated_received_symbol)

    current_symbol_start_location += (L+N)

  return estimated_received_constellation_symbols

# Functions for estimating noise variance
def calculate_noise_variance(H, known_OFDM_symbol, received_signal, known_OFDM_block_start_index, known_OFDM_block_end_index, n):
  # Isolate the received known OFDM symbol block in the time domain signal.
  received_known_OFDM_symbol_block_time_domain = received_signal[known_OFDM_block_start_index:known_OFDM_block_end_index]

  # Loop through the received known OFDM symbols to get an estimate of the channel frequency response for each symbol.
  # Split into arrays containing each OFDM symbol in the time domain in its own array.
  array_of_individual_known_OFDM_symbols_time_domain = np.split(received_known_OFDM_symbol_block_time_domain, n)

  # Remove the cyclic prefixes.
  array_of_individual_known_OFDM_symbols_time_domain_noCP = [removeCP(OFDM_symbol_time_domain_with_CP) for OFDM_symbol_time_domain_with_CP in array_of_individual_known_OFDM_symbols_time_domain]

  # Take the DFT of each known OFDM symbol in the time domain. i.e. demodulate.
  array_of_individual_known_OFDM_symbols_frequency_domain = [DFT(OFDM_symbol_time_domain) for OFDM_symbol_time_domain in array_of_individual_known_OFDM_symbols_time_domain_noCP]

  # Keep only the relevant frequency bins. i.e. 1 to 1023. This is Y.
  array_of_individual_known_OFDM_symbols_frequency_domain_relevant_half = [OFDM_symbol_frequency_domain[1:int(N/2)] for OFDM_symbol_frequency_domain in array_of_individual_known_OFDM_symbols_frequency_domain]

  # Multiply each originall transmitted known OFDM symbol by H. This is HX.
  received_symbol_without_noise = H*known_OFDM_symbol

  # Estimate the noise. This is N = Y - HX.
  noise_estimates = [(Y - received_symbol_without_noise) for Y in array_of_individual_known_OFDM_symbols_frequency_domain_relevant_half]

  noise_estimates = np.array(noise_estimates)

  noise_estimates = noise_estimates.flatten()

  # Get real and imaginary parts of noise N.
  noise_estimates_real = noise_estimates.real
  noise_estimates_imaginary = noise_estimates.imag

  # Compute the element by element noise variances along the real and imaginary axis.
  noise_variances_real = noise_estimates_real*noise_estimates_real
  noise_variances_imaginary = noise_estimates_imaginary*noise_estimates_imaginary

  # Compute the mean noise variances along the real and imaginary axis.
  mean_noise_variance_real = np.mean(noise_variances_real)
  mean_noise_variance_imaginary = np.mean(noise_variances_imaginary)

  return mean_noise_variance_real, mean_noise_variance_imaginary

# # Decoding using LDPC.
# Defining all the neccessary functions for LDPC
def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr

def bits2bytes(x):
    n = len(x)+3
    r = (8 - n % 8) % 8
    prefix = format(r, '03b')
    x = ''.join(str(a) for a in x)
    suffix = '0'*r
    x = prefix + x + suffix
    x = [x[k:k+8] for k in range(0,len(x),8)]
    y = []
    for a in x:
        y.append(int(a,2))

    return y

def bytes2bits(y):
    x = [format(a, '08b') for a in y]
    r = int(x[0][0:3],2)
    x = ''.join(x)
    x = [int(a) for a in x]
    for k in range(3):
        x.pop(0)
    for k in range(r):
        x.pop()
    return x

def mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])

def SP(bits):
    return bits.reshape((int(len(bits)/2), 2))

def encode_source_bits(source_bits):
    tem = []
    for i in range(int(len(source_bits)//c.K)):
        tem.extend(c.encode(source_bits[i*c.K:(i+1)*c.K]))
    return np.array(list(tem))

def LLR_calculation(output, channel_frequency, noise_variance):
    #output = np.array(list(output))
    #channel_frequency = [i[1+drop_front:N//2-drop_back] for i in channel_frequency]
    #channel_frequency = np.concatenate(channel_frequency, axis=None)[:len(output)]
    LLR = []

    for i in range(len(output)):
        first = channel_frequency[i]*np.conjugate(channel_frequency[i])*np.sqrt(2)*np.imag(output[i])/noise_variance
        second = channel_frequency[i]*np.conjugate(channel_frequency[i])*np.sqrt(2)*np.real(output[i])/noise_variance
        LLR.append(np.real(first))
        LLR.append(np.real(second))
    return np.array(LLR)

def BER(input_bits, output_bits):
    added = (input_bits + output_bits) % 2
    return np.sum(added)/len(input_bits)

def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def frombits(bits):
    chars = []
    for b in range(int(len(bits) / 8)):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def s_to_bitlist(s):
    ords = (ord(c) for c in s)
    shifts = (7, 6, 5, 4, 3, 2, 1, 0)
    return [(o >> shift) & 1 for o in ords for shift in shifts]
def bitlist_to_chars(bl):
    bi = iter(bl)
    bytes = zip(*(bi,) * 8)
    shifts = (7, 6, 5, 4, 3, 2, 1, 0)
    for byte in bytes:
        yield chr(sum(bit << s for bit, s in zip(byte, shifts)))
def bitlist_to_s(bl):
    return ''.join(bitlist_to_chars(bl))

def convert_file_to_bits(filename):
    """Takes a file associated with the filename and converts it first to bytes, then to a bit stream"""
    # file is txt file
    with open(filename, "r") as text:
        text_string = text.read()
    return s_to_bitlist(text_string)

def LDPC_decoding(data_constellation_estimated, H_est):
    output = data_constellation_estimated
    # Decoder
    c = ldpc.code()
    channel_frequency = np.repeat(H_est, len(output)/len(H_est))
    output = LLR_calculation(output,channel_frequency,noise_variance)
    tem = []
    for i in range(int(len(output)//(c.K*2))):
        app,it = c.decode(output[c.K*2*i:c.K*2*i+2*c.K])
        #tem.extend(app[::2])
    #         app = output[c.K*2*i:c.K*2*i+2*c.K]
        tem.extend(app[:int(len(app)/2)])

    output = []
    for i in range(len(tem)):
        if tem[i]<=0:
            output.append(1)
        elif tem[i]>0:
            output.append(0)

    output_bits = output
    output = bitarray(output)
    return output_bits

def simulate_received_signal():
    transmitted_signal = convert_wav_to_array('audio_to_transmit.wav')

    channelResponse = pd.read_csv('channel.csv')
    zero_array = np.zeros(len(transmitted_signal) - len(channelResponse))
    channelResponse = np.append(channelResponse, zero_array)

    received_signal = signal.convolve(channelResponse, transmitted_signal, mode='full')

    return received_signal


received_signal = simulate_received_signal()

chirp_signal = generateChirp(chirp_constants, sample_rate)
chirp_center_index_in_received_signal = matched_filter(received_signal, chirp_signal)
delay = chirp_center_index_in_received_signal - len(chirp_signal)/2
chirp_length = len(chirp_signal)
chirp_start_index = chirp_center_index_in_received_signal - int((chirp_length / 2))
chirp_end_index = chirp_center_index_in_received_signal + int((chirp_length / 2))

known_OFDM_symbol_block_start_location = chirp_end_index
known_OFDM_symbol_block_end_location = known_OFDM_symbol_block_start_location + (n*(L+N))
OFDM_data_symbol_block_start_location = known_OFDM_symbol_block_end_location

H_est, h_est = estimate_channel(known_OFDM_symbol_block_start_location, known_OFDM_symbol_block_end_location, received_signal, N, L, n)

number_of_OFDM_data_blocks = data_length//(N+L)
data_constellation_estimated = demodulate(OFDM_data_symbol_block_start_location, number_of_OFDM_data_blocks, received_signal, N, L)
output_bits = LDPC_decoding(data_constellation_estimated, H_est)


input_bits = convert_file_to_bits('data_text.txt')
output_bits = np.array(output_bits)[:len(input_bits)]

#print(bitlist_to_s(output_bits)) #uncommenting this should print the decoded data text

print(BER(input_bits, output_bits))
