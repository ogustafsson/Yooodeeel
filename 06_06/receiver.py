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
from sklearn.linear_model import LinearRegression # Module for performing linear regression.
import math

import ldpc
from bitarray import bitarray
from PIL import Image
import binascii

# Define Constants.
N = 2048 #OFDM symbol length
block_size = N//2 - 1 # Set block size in frequency domain to half the OFDM symbol subtract the zeroed first frequency bin
L = 256 #Cyclic prefix length
n = 10 # Number of repetitions of known OFDM symbol for transmission
sample_rate = 44100 # Sample rate is 44.1 kHz
FILE_NAME = 'output.wav'
# Define the QPSK constellation and its power
A = np.sqrt(2) # Radius of the QPSK constellation circle
chirp_constants = {"startFrequency": 100, "endFrequency": 10000, "duration": 1, "method": "logarithmic"}
#Defining constants and mapping/demapping tables


bit_mapping = {(0,0): 0, (0,1): 1, (1,1): 2, (1,0): 3}# Use Gray coding to convert bit stream to constellation indices
mapping_table = {(0,0): (1+1j), (0,1): (-1+1j), (1,1): (-1-1j), (1,0): (1-1j)}
constellation_mapping = A/np.sqrt(2) * np.array([(1+1j), (-1+1j), (-1-1j), (1-1j)]) # Define the mapping of indices to QPSK constellation in anti-clockwise sense


noise_variance = 1

c = ldpc.code(z = 81)
k = c.K #data bits need to be a multiple of k in order to do LDPC encoding/Decoding



# TO-DO: New Constants.
pilot_tones_start_index = 1
pilot_tones_end_index = 1018
pilot_tones_step = 8
first_frequency_bin = 49
last_frequency_bin = 699




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

# Returns only the real part of the iDFT.
def iDFT_real(signal):
  inverse_transform = ifft(signal)
  return inverse_transform.real
  
def SP(bits):
    return bits.reshape((len(bits)//2, 2))
def Mapping(bits):
    bits = SP(bits)
    return np.array([mapping_table[tuple(b)] for b in bits])

    
# Generate the chirp signal.
def generate_chirp():
   """Produces exponential chirp with exponential envelope"""
   fs = sample_rate
   chirp_length = chirp_constants['duration']
   f1 = chirp_constants['startFrequency']
   f2 = chirp_constants['endFrequency']
   window_strength = 50

   T = chirp_length
   t = np.linspace(0, T, T * fs)
   r = f2/f1

   # Calculate Sine Sweep time domain values
   profile = np.sin(2*np.pi*T*f1*((r**(t/T)-1)/(np.log(r))))*(1-np.e**(-window_strength*t))*(1-np.e**(window_strength*(t-T)))

   return profile

# Matched filter returns index of highest correlation between a noisy signal and the desired signal.
# For the chirp, this index will be located halfway along the chirp.
def matched_filter(noisy_signal, desired_signal):
    desired_signal = desired_signal.flatten()
    correlation = signal.correlate(noisy_signal, desired_signal, mode='same')
    max_correlation = np.amax(correlation)
    max_correlation_index = np.argmax(correlation)

    return max_correlation_index

# Remove cyclic prefix
def removeCP(signal):
    """Remove the cyclic prefix of a signal"""
    return signal[L:(L+N)]


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

def generate_random_bit_sequence(bit_sequence_length, random_seed):
    #random seed is the number corresponding to the random seed, aka 2020, 2021, 2022, etc
    rng = np.random.default_rng(random_seed)
    random_bit_sequence = rng.integers(low=0, high=2, size = bit_sequence_length)
    return random_bit_sequence

# Note that bits should be in numpy array like [0,0,0,1,1,0,…]
def map_bits_to_constellation_indices(bit_stream):
    """Takes a stream of bits and maps them to constellation indices from range {0,1,2,3} using Gray coding"""
    bit_stream_length=len(bit_stream) # Compute length of bit stream
    # If there is an odd number of bits, add a zero bit on the end to make it even
    if bit_stream_length % 2:
        bit_stream = np.append(bit_stream, np.array([0]))
    bit_pairs = np.split(bit_stream, len(bit_stream)//2) # Split bit stream array into sub-arrays of bit pairs like [[0,0], [0,1], …]

    constellation_indices = np.array([]) # Set up empty array for the loop
    # Map each bit pair to its corresponding constellation index
    for bit_pair in bit_pairs:
        constellation_indices = np.append(constellation_indices, bit_mapping[tuple(bit_pair)]) # TODO: make this loop more efficient

    constellation_indices = constellation_indices.astype(int) # Ensure that constellation indices are integers

    return constellation_indices

def map_indices_to_constellation(constellation_indices):
    """Takes sequence of constellation indices from range {0,1,2,3} and maps them to a sequence of QPSK constellation symbols"""

    # UNIT TEST: check constellation indices belong to range {0,1,2,3}
    assert not np.isin([False], np.isin(constellation_indices, [0,1,2,3])), "Indices do not belong to range {0,1,2,3}"

    constellation_sequence = np.array([]) # Set up empty array for the loop
    # Map each random index to its corresponding constellation symbol
    for index in constellation_indices:
        constellation_sequence = np.append(constellation_sequence, constellation_mapping[index]) # TODO: make this loop more efficient

    return constellation_sequence


# Generate random (known) OFDM symbols in the frequency domain.
def create_random_OFDM_symbols_frequency(number_of_blocks, random_seed):
    """
    Input: number of blocks of length N = 2048 we want to generate all at once (number_of_blocks = 1 for seed_2020, number_of_blocks = 5 for seed_2021, number_of_blocks = 1 for seed_2022)
    Output: number_of_blocks*(N+L) time domain long sequence
    """
    bit_sequence_length = number_of_blocks*(N-2)
    random_bit_sequence = generate_random_bit_sequence(bit_sequence_length, random_seed)
    blocks = Mapping(random_bit_sequence)
    block_size = N//2 - 1
    blocks = np.split(blocks, len(blocks)/block_size)

    return blocks

# Returns H_est and h_est by performing channel estimation using known OFDM symbols.
def estimate_channel(known_OFDM_symbol_block_start_location, known_OFDM_symbol_block_end_location, received_signal, N, L, n):
  # Generate the known OFDM symbol.
  # TO-DO: Needs to be updated to generate the new version of the known OFDM symbol. Or a list of known OFDM symbols if we use several.
  known_OFDM_symbols = create_random_OFDM_symbols_frequency(5, 2021)
  # TO-DO: Be careful!!!!!!

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
  # Here we loop through all of the demodulated frequency domain arrays and divide by the appropriate known OFDM symbol.
  # TO-DO: Needs to be updated if several known OFDM symbols are used instead of 1. Would need to use a slightly more elaborate loop.
  known_OFDM_symbol_index = 0
  array_of_frequency_response = np.array([])
  for OFDM_demod in array_of_individual_known_OFDM_symbols_frequency_domain_relevant_half:
      current_frequency_response = OFDM_demod / known_OFDM_symbols[known_OFDM_symbol_index]
      array_of_frequency_response = np.append(array_of_frequency_response, current_frequency_response)
      if (known_OFDM_symbol_index == 4):
          known_OFDM_symbol_index = 0
      else:
          known_OFDM_symbol_index += 1

  # Split the array so that the average can be computed.
  array_of_frequency_response = np.split(array_of_frequency_response, n)

  # Compute the averaged estimated frequency response.
  # TO-DO: Check if this averaging works!!! Be careful!!!
  H_est = compute_average(array_of_frequency_response, int(N/2 - 1), n)

  # To get the time domain impulse response (to be real), we need to have conjugate symmetry in the frequency domain. With zeros in the right places.
  reversed_H_est = np.conj(np.append(np.array([0]), np.flip(H_est)))
  H_est_symmetric = np.append(H_est, reversed_H_est)
  H_est_symmetric = np.append(np.array([0]), H_est_symmetric)

  # Compute the impulse response.
  h_est = iDFT_real(H_est_symmetric)

  return H_est, h_est


def demodulate_pilot_tones(H, OFDM_data_symbol_block_start_location, number_of_OFDM_data_blocks, received_signal, N, L, pilot_tones_start_index, pilot_tones_end_index, pilot_tones_step, first_frequency_bin, last_frequency_bin, update_sync_using_pilots=True):
  estimated_received_data_constellation_symbols = np.array([])
  estimated_received_pilot_constellation_symbols = np.array([])
  pilot_indices_overall = np.array([])

  current_symbol_start_location = OFDM_data_symbol_block_start_location
  timing_offsets = np.array([])

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
    estimated_received_symbol = current_symbol_frequency_domain_relevant_part / H
    
    # TO-DO: Modify the pilot tone indices and the data indices to be such that they correspond to the
    # frequency bins with pilot tones and data are located due to the standard.
    # Create arrays containing the pilot tone indices and the data tone indices.
    all_indices = np.arange(N//2-1)
    pilot_indices = np.arange(pilot_tones_start_index, pilot_tones_end_index, pilot_tones_step)
    # Delete the pilot tone indices.
    data_indices = np.delete(all_indices, pilot_indices)
    # Remove the lower and upper frequency bins from the data indices.
    data_indices_truncated = []
    for index in data_indices:
        if (index >= first_frequency_bin and index<last_frequency_bin):
            data_indices_truncated.append(index)

    data_indices_truncated = np.array(data_indices_truncated)


    # Extract the pilot tones and data tones from the estimated received symbol by taking the values at the appropriate indices.
    estimated_received_pilot_tones = np.take(estimated_received_symbol, pilot_indices)
    estimated_received_data_tones = np.take(estimated_received_symbol, data_indices_truncated)

    # If this parameter is true, then the synchronisation for each OFDM data symbol is update using the information from the pilot tones.
    if (update_sync_using_pilots == True):
        # Update the synchronisation of each OFDM data symbol here by rotating each constellation symbol by the appropriate amount.
        pilot_tone = A/np.sqrt(2)*(1+1j)
        phase_shifts = np.angle(estimated_received_pilot_tones / pilot_tone)
        # TO-DO: Modify which frequency bins to consider for this part.
        phase_shifts_unwrapped = np.unwrap(phase_shifts)[10:60] # Here we take a section of the unwrapped phases. Investigate further.
        adjusted_pilot_indices = (2*np.pi/N)*pilot_indices[10:60] # Here we take a section of the unwrapped phases. Investigate further.
        
        # Fit linear regression to the unwrapped phase shifts, to determine the offset in number of samples.
        model = LinearRegression().fit(adjusted_pilot_indices[:, np.newaxis], phase_shifts_unwrapped)
        slope = model.coef_[0]

        # Round the offset to two decimal places and add it to a list.
        offset = np.round(slope, 2)
        timing_offsets = np.append(timing_offsets, offset)

        # Correct for the timing offset by rotating the estimated received constellation symbols.
        # This is done by multiplying the received data constellation symbols by a complex exponential.
        adjusted_data_indices_truncated = (2*np.pi/N)*data_indices_truncated
        resynchronisation_multiplier = np.exp((-1j)*(offset)*adjusted_data_indices_truncated)
        estimated_received_data_tones = estimated_received_data_tones*resynchronisation_multiplier


    # Append estimated received symbol to array of all received constellation symbols.
    estimated_received_data_constellation_symbols = np.append(estimated_received_data_constellation_symbols, estimated_received_data_tones)
    estimated_received_pilot_constellation_symbols = np.append(estimated_received_pilot_constellation_symbols, estimated_received_pilot_tones)
    pilot_indices_overall = np.append(pilot_indices_overall, pilot_indices)
    current_symbol_start_location += (L+N)


  return estimated_received_data_constellation_symbols, estimated_received_pilot_tones, pilot_indices_overall



# Functions for LDPC decoding and handling output bits.
# # Decoding using LDPC.
# Defining all the neccessary functions for LDPC decoding.
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

def convert_file_to_bits(file_name):
    with open(file_name, 'rb') as file:
        file_bytes = np.array([byte for byte in file.read()], dtype=np.uint8)
    file_bits = np.unpackbits(file_bytes)
    file_bits = np.array(file_bits, dtype=int)
    return file_bits
def write_file(binary_data, file_type):
    data_bytes = np.packbits(binary_data)
    data_bytes = bytearray(data_bytes)
    with open('output' + file_type, 'wb') as file:
        file.write(data_bytes)


def LDPC_decoding(data_constellation_estimated, H_est):
    output = data_constellation_estimated
    # Decoder

    channel_frequency = np.repeat(H_est, len(output)//len(H_est) + 1)

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

# Demapping functions using maximum likelihood
def demapping_symbol_ML(symbol):
    first_bit = 0 if symbol.imag >= 0 else 1
    second_bit = 0 if symbol.real >= 0 else 1
    return first_bit, second_bit

def demapping_ML(sequence):
    output = []
    for f in sequence:
        first_bit, second_bit = demapping_symbol_ML(f)
        output.append(first_bit)
        output.append(second_bit)
    return output

def calculate_amount_of_padding_for_ldpc(source_bits_length):
    additional_bits = 0
    if (source_bits_length % k != 0):
        additional_bits = k - source_bits_length % k
    
    return additional_bits

# Calculate the file length from the repeated received 32 file_length_bits.
def calculate_file_length(file_length_bits):
    file_length_blocks = np.split(file_length_bits, 5)
    # Sum the blocks to perform majority vote.
    vote_array = np.zeros(32)
    for block in file_length_blocks:
        vote_array += block
    
    # Now perform the majority vote to return the estimate of the transmitted bits.
    corrected_file_length_bits = []
    for element in vote_array:
        if element > 2:
            corrected_file_length_bits.append(1)
        else:
            corrected_file_length_bits.append(0)
    
    # Convert the file length from binary to an integer.
    bits = bitarray(corrected_file_length_bits)
    i = 0
    for bit in bits:
        i = (i << 1) | bit
    
    file_length = i

    return file_length

# Obtain the file type by performing a majority vote and then finding the minimum Hamming distance codeword.
# Returns 0 if .tif, 1 if .txt, and 2 if .wav.
def obtain_file_type(file_type_bits):
    file_type_blocks = np.split(file_type_bits, 5)
    # Sum the blocks to perform majority vote.
    vote_array = np.zeros(8)
    for block in file_type_blocks:
        vote_array += block
    
    # Now perform the majority vote to return the estimate of the transmitted bits.
    voted_file_type_bits = []
    for element in vote_array:
        if element > 2:
            voted_file_type_bits.append(1)
        else:
            voted_file_type_bits.append(0)
    
    estimated_file_type_bits = np.array(voted_file_type_bits)

    # Now perform Hamming distance calculations to determine the transmitted codeword.
    # TO-DO: These should probably be global variables.
    tif_codeword = np.array([0,0,1,1,0,0,1,0]) # Index 0
    txt_codeword = np.array([0,1,0,0,1,0,0,0]) # Index 1
    wav_codeword = np.array([1,0,1,0,0,1,0,1]) # Index 2

    tif_hamming_distance = np.sum((estimated_file_type_bits + tif_codeword) % 2)
    txt_hamming_distance = np.sum((estimated_file_type_bits + txt_codeword) % 2)
    wav_hamming_distance = np.sum((estimated_file_type_bits + wav_codeword) % 2)

    hamming_distances = np.array([tif_hamming_distance, txt_hamming_distance, wav_hamming_distance])

    min_index = np.argmin(hamming_distances)

    return min_index
    



def simulate_received_signal_pilot_tones():
    transmitted_signal = convert_wav_to_array('audio_to_transmit.wav')

    channelResponse = pd.read_csv('channel.csv', header = None)
    channelResponse = np.array(channelResponse)
    channelResponse = channelResponse.flatten()
    zero_array = np.zeros(len(transmitted_signal) - len(channelResponse))
    channelResponse = np.append(channelResponse, zero_array)

    received_signal = signal.convolve(channelResponse, transmitted_signal, mode='full')

    return received_signal


# Load the received signal.
received_signal = convert_wav_to_array('inputs.wav') # TO-DO: Scan in the received signal.
#received_signal = simulate_received_signal_pilot_tones()

# Perform chirp synchronisation.
# TO-DO: Update the function generating the chirp to take the analytical method of chirp generation.
# TO-DO: Take into account the chirp at the end of the transmission.
chirp_signal = generate_chirp()
# Matched filtering returns the center index of the chirp in the received signal.
# TO-DO: Think about chirp. This should no longer be an issue now that the chirp at the end of the signal is reversed.
chirp_center_index_in_received_signal = matched_filter(received_signal, chirp_signal)
# Now compute the start and end positions of the chirp signal.
chirp_length = len(chirp_signal)
chirp_start_index = chirp_center_index_in_received_signal - int((chirp_length / 2))
chirp_end_index = chirp_center_index_in_received_signal + int((chirp_length / 2))

# Compute the start and end locations of the known OFDM symbol block.
known_OFDM_symbol_block_start_location = chirp_end_index + (L+N)
known_OFDM_symbol_block_end_location = known_OFDM_symbol_block_start_location + (n*(L+N))


# TO-DO: implement the modified Schmidl and Cox synchronisation.

#Plot the received signal.
"""
x = list(range(0, len(received_signal)))
x = np.array(x)
fig, ax = plt.subplots()
plt.title("Received Signal")
plt.xlabel("Sample")
plt.ylabel("Signal Magnitude")
plt.axvline(x=chirp_start_index, color='green') #Should be at beginning of chirp
plt.axvline(x=chirp_end_index, color='red') #Should be at end of chirp which is at chirp beginning + chirp_duration*44100
#plt.xlim(0,4000 )
# plt.ylim(-0.2, 0.2)
ax.plot(x, received_signal)
plt.legend(['Estimated Chirp Start', 'Estimated Chirp End', 'Signal'])
plt.show()
"""

# Perform the initial channel estimation based on this synchonisation point.
# To-consider: maybe we use 10 times repeated the same OFDM symbol, or 10 different symbols?
H_est, h_est = estimate_channel(known_OFDM_symbol_block_start_location, known_OFDM_symbol_block_end_location, received_signal, N, L, n)

# Plot the estimated frequency response and impulse response of the channel.
# Magnitude Frequency Response.
"""
f = list(range(1, len(H_est)+1))
frequencies = np.array(f)
frequencies = (sample_rate / N) * frequencies
x = frequencies
fig, ax = plt.subplots()
plt.title('Estimated Magnitude Frequency Response (Known OFDM symbols)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Magnitude')
plt.yscale('log')
ax.plot(x, abs(H_est))
plt.show()
# Phase Frequency Response
f = list(range(1, len(H_est)+1))
frequencies = np.array(f)
frequencies = (sample_rate / N) * frequencies
x = frequencies
fig, ax = plt.subplots()
plt.title('Estimated Phase Frequency Response (Known OFDM symbols)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (rad)')
#plt.yscale('log')
ax.plot(x, np.angle(H_est))
plt.show()
# Impulse Response
t = list(range(1, len(h_est)+1))
times = np.array(t)
#times = times / sample_rate
x = times
fig, ax = plt.subplots()
plt.title("Estimated Channel Impulse Response")
plt.xlabel("Sample")
plt.ylabel("Signal Magnitude")
ax.plot(x, h_est)
plt.show()
"""

# Perform dynamic phase adjustment to update the initial synchronisation point, and update the channel estimates.
# Get the phase response and unwrap it.
phase_response = np.angle(H_est)
phase_response_unwrapped = np.unwrap(phase_response)

frequency_bin_indices = np.array(list(range(1, len(H_est)+1)))
adjusted_frequency_bin_indices = (2*np.pi/N)*frequency_bin_indices

# Look at only the middle part of the phase frequency response. As this seems to normally be the linear part?
# TO-DO: Consider more carefully which part of the phase response should be considered.
phase_response_unwrapped_truncated = phase_response_unwrapped[300:800]
adjusted_frequency_bin_indices_truncated = adjusted_frequency_bin_indices[300:800]

# Fit a linear regression to determine how much off synchronisation we are.
model_phase_response = LinearRegression().fit(adjusted_frequency_bin_indices_truncated[:, np.newaxis], phase_response_unwrapped_truncated)
slope_phase_response = model_phase_response.coef_[0]
# Round the slope to the nearest integer.
sync_offset = int(np.round(slope_phase_response))

# Reset the synchronisation point based on this new information.
known_OFDM_symbol_block_start_location_updated = known_OFDM_symbol_block_start_location - sync_offset
known_OFDM_symbol_block_end_location_updated = known_OFDM_symbol_block_end_location - sync_offset

# Now repeat channel estimation with this updated synchronisation point.
H_est_updated, h_est_updated = estimate_channel(known_OFDM_symbol_block_start_location_updated, known_OFDM_symbol_block_end_location_updated, received_signal, N, L, n)

# Create an array to hold the all the estimated data constellation symbols.
data_constellation_estimated = np.array([])


# Demodulate the first payload block to obtain the header metadata.
# TO-DO: For now, we are assuming that at least 1 complete payload data block is transmitted.
first_payload_block_start_location = known_OFDM_symbol_block_end_location_updated + (L+N)
first_payload_block_demodulated_data_symbols, first_payload_block_demodulated_pilot_symbols, first_payload_block_pilot_indices_overall = demodulate_pilot_tones(H_est_updated, first_payload_block_start_location, 10, received_signal, N, L, pilot_tones_start_index, pilot_tones_end_index, pilot_tones_step, first_frequency_bin, last_frequency_bin, update_sync_using_pilots=True)
data_constellation_estimated = np.append(data_constellation_estimated, first_payload_block_demodulated_data_symbols)

# Obtain the header metadata from the first payload block.


# Demodulate and decode the constellation symbols corresponding to the first LDPC block (k=972 and n=1944 for LDPC code since rate is 1/2).
first_ldpc_block_constellation_symbols = data_constellation_estimated[:972]
first_ldpc_block_decoded = LDPC_decoding(first_ldpc_block_constellation_symbols, H_est_updated)

# Extract the metadata bits.
metadata_bits = np.array(first_ldpc_block_decoded[:200])

file_type_bits = metadata_bits[:40]
print(file_type_bits[8:16])
file_length_bits = metadata_bits[40:]
print(file_length_bits[:32])

# Obtain the file size from the repeated file length bits.
file_size = calculate_file_length(file_length_bits)

# Obtain the file type.
# 0: tif, 1: txt, 2: wav
file_type = obtain_file_type(file_type_bits)

print("File size in bits: " + str(file_size))
print("File type as index: " + str(file_type))


# Calculate the total number of OFDM data symbols.
# Could use the second chirp to verify this calculation.
# TO-DO: Consider the effect of padding on the total number of constellation symbols.

all_indices = np.arange(N//2-1)
pilot_indices = np.arange(pilot_tones_start_index, pilot_tones_end_index, pilot_tones_step)
# Delete the pilot tone indices.
data_indices = np.delete(all_indices, pilot_indices)
# Remove the lower and upper frequency bins from the data indices.
data_indices_truncated = []
for index in data_indices:
    if (index >= first_frequency_bin and index<last_frequency_bin):
        data_indices_truncated.append(index)

data_indices_truncated = np.array(data_indices_truncated)

print("The number of data indices is: " + str(len(data_indices_truncated)))


padding = calculate_amount_of_padding_for_ldpc(file_size)
total_number_of_data_constellation_symbols = file_size + padding # TO-DO: Need to consider LDPC rate and 2 bits per constellation symbol. Consider padding.
total_number_of_OFDM_data_symbols = int(np.ceil(total_number_of_data_constellation_symbols / len(data_indices_truncated))) # TO-DO: Think about how to do this calculation.
total_number_of_complete_payload_blocks = total_number_of_OFDM_data_symbols // 10 # TO-DO: Check this.
number_of_OFDM_symbols_in_incomplete_payload_block = total_number_of_OFDM_data_symbols % 10


print("total number of OFDM data symbols is: " + str(total_number_of_OFDM_data_symbols))
print("Number of complete payload blocks is: " + str(total_number_of_complete_payload_blocks))
print("number of OFDM symbols in incomplete_payload_block is: " + str(number_of_OFDM_symbols_in_incomplete_payload_block))

# Demodulate the data to produce an array of data constellation symbols.
# Need to loop over the repeating structure.
# Starting from the second payload block.
# The payload block length is 11.
# First we demodulate all the completely filled payload blocks.

# Test this part by decoding payload block by payload block.
# TO-DO: Check that the correct number of OFDM symbols is actually getting decoded.
current_payload_block_data_start_location = first_payload_block_start_location + (11*(L+N))
for i in range(total_number_of_complete_payload_blocks):
    current_payload_block_demodulated_data_symbols, current_payload_block_demodulated_pilot_symbols, current_payload_block_pilot_indices_overall = demodulate_pilot_tones(H_est_updated, current_payload_block_data_start_location, 10, received_signal, N, L, pilot_tones_start_index, pilot_tones_end_index, pilot_tones_step, first_frequency_bin, last_frequency_bin, update_sync_using_pilots=True)
    data_constellation_estimated = np.append(data_constellation_estimated, current_payload_block_demodulated_data_symbols)
    current_payload_block_data_start_location += (11*(L+N))

if (number_of_OFDM_symbols_in_incomplete_payload_block != 0):
    current_payload_block_demodulated_data_symbols, current_payload_block_demodulated_pilot_symbols, current_payload_block_pilot_indices_overall = demodulate_pilot_tones(H_est_updated, current_payload_block_data_start_location, number_of_OFDM_symbols_in_incomplete_payload_block, received_signal, N, L, pilot_tones_start_index, pilot_tones_end_index, pilot_tones_step, first_frequency_bin, last_frequency_bin, update_sync_using_pilots=True)
    data_constellation_estimated = np.append(data_constellation_estimated, current_payload_block_demodulated_data_symbols)


# Recover the output bits through LDPC decoding.
output_bits = LDPC_decoding(data_constellation_estimated, H_est_updated)
# TO-DO: Keep only the output bits that correspond to the actual transmitted data, and not the the other random symbols.
# TO-DO: Remove header metadata from beginning of output bits.
output_bits = np.array(output_bits[200:file_size+200])


# Compare output and input bits, and view the output.
input_bits = convert_file_to_bits('data_text.txt')
print(bitlist_to_s(output_bits)) #uncommenting this should print the decoded data text
print(BER(input_bits, output_bits))

# TO-DO: Need to add code for dealing with other file types except txt.
