import numpy as np
import csv
import pandas as pd
from scipy.linalg import dft
from scipy.fft import fft
from scipy.fft import ifft
import matplotlib.pyplot as plt
import bitarray
import pdb
from scipy.io.wavfile import write # For writing to a .wav file
from scipy.io.wavfile import read # For reading from a .wav file to an np array.
from scipy import signal # Module for generating chirp.
from sklearn.linear_model import LinearRegression # Module for performing linear regression.

import ldpc
from bitarray import bitarray
from PIL import Image
import binascii

# Define Constants.
N = 2048 #OFDM symbol length
L = 256 #Cyclic prefix length
n = 10 # Number of repetitions of known OFDM symbol for transmission
sample_rate = 44100 # Sample rate is 44.1 kHz
FILE_NAME = 'output.wav'
# Define the QPSK constellation and its power
A = 10 # Radius of the QPSK constellation circle
chirp_constants = {"startFrequency": 100, "endFrequency": 10000, "duration": 1, "method": "logarithmic"}
#Defining constants and mapping/demapping tables
mapping_table = {
    (0,0) : A*(1+1j),
    (0,1) : A*(-1+1j),
    (1,1) : A*(-1-1j),
    (1,0): A*(1-1j)
}
demapping_table = {v : k for k, v in mapping_table.items()}
noise_variance = 1


# TO-DO: New Constants.
pilot_tones_start_index = 1
pilot_tones_end_index = 1018
pilot_tones_step = 8
first_frequency_bin = 50
last_frequency_bin = 700




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

# Generate chirp signal
# TO-DO: Update the function generating the chirp to take the analytical method of chirp generation.
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

# Returns H_est and h_est by performing channel estimation using known OFDM symbols.
def estimate_channel(known_OFDM_symbol_block_start_location, known_OFDM_symbol_block_end_location, received_signal, N, L, n):
  # Generate the known OFDM symbol.
  # TO-DO: Needs to be updated to generate the new version of the known OFDM symbol. Or a list of known OFDM symbols if we use several.
  known_OFDM_symbol = # Function for generating the known OFDM symbols.
  known_OFDM_symbols = # Sub-arrays containing each of the 5 known OFDM symbols.

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


def simulate_received_signal_pilot_tones():
    transmitted_signal = convert_wav_to_array('pilot_audio_to_transmit.wav')

    channelResponse = pd.read_csv('channel.csv', header = None)
    channelResponse = channelResponse.flatten()
    zero_array = np.zeros(len(transmitted_signal) - len(channelResponse))
    channelResponse = np.append(channelResponse, zero_array)

    received_signal = signal.convolve(channelResponse, transmitted_signal, mode='full')

    return received_signal


# Load the received signal.
received_signal = # TO-DO: Scan in the received signal.

# Perform chirp synchronisation.
# TO-DO: Update the function generating the chirp to take the analytical method of chirp generation.
chirp_signal = generateChirp(chirp_constants, sample_rate)
# Matched filtering returns the center index of the chirp in the received signal.
chirp_center_index_in_received_signal = matched_filter(received_signal, chirp_signal)
# Now compute the start and end positions of the chirp signal.
chirp_length = len(chirp_signal)
chirp_start_index = chirp_center_index_in_received_signal - int((chirp_length / 2))
chirp_end_index = chirp_center_index_in_received_signal + int((chirp_length / 2))

# Compute the start and end locations of the known OFDM symbol block.
known_OFDM_symbol_block_start_location = chirp_end_index + (L+N)
known_OFDM_symbol_block_end_location = known_OFDM_symbol_block_start_location + (n*(L+N))

# To-do: implement the modified Schmidl and Cox synchronisation.

# Perform the initial channel estimation based on this synchonisation point.
# To-consider: maybe we use 10 times repeated the same OFDM symbol, or 10 different symbols?
H_est, h_est = estimate_channel(known_OFDM_symbol_block_start_location, known_OFDM_symbol_block_end_location, received_signal, N, L, n)

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
first_payload_block_start_location = known_OFDM_symbol_block_end_location_updated + (L+N)
first_payload_block_demodulated_data_symbols, first_payload_block_demodulated_pilot_symbols, first_payload_block_pilot_indices_overall = demodulate_pilot_tones(H_est_updated, first_payload_block_start_location, 10, received_signal, N, L, pilot_tones_start_index, pilot_tones_end_index, pilot_tones_step, first_frequency_bin, last_frequency_bin, update_sync_using_pilots=True)
data_constellation_estimated = np.append(data_constellation_estimated, first_payload_block_demodulated_data_symbols)

# Obtain the header metadata from the first payload block.
# Extract the constellation symbols corresponding to the metadata header.
# TO-DO: Need to know what the upper value is in this array splice.
metadata_constellation_symbols = data_constellation_estimated[:]



# Take the constellation symbols that correspond to the header metadata.
# Convert them to bits.
# Obtain the file_type (ASCII string) and file_size(number of bits as integer).
file_type = # Something e.g. '.tif'
file_size = # Some number of bits.






# TO-DO: Remove the constellation symbols related to the header metadata. 
# TO-DO: Need to find the lower value of this array splice.
data_constellation_estimated = data_constellation_estimated[:]

# Calculate the total number of OFDM data symbols.
# Could use the second chirp to verify this calculation.
total_number_of_data_constellation_symbols = file_size # TO-DO: Need to consider LDPC rate and 2 bits per constellation symbol.
total_number_of_OFDM_data_symbols = np.ceil(total_number_of_data_constellation_symbols / (last_frequency_bin - first_frequency_bin)) # TO-DO: Think about how to do this calculation.
total_number_of_complete_payload_blocks = total_number_of_OFDM_data_symbols // 10 # TO-DO: Check this.
number_of_OFDM_symbols_in_incomplete_payload_block = total_number_of_OFDM_data_symbols % 10




# Demodulate the data to produce an array of data constellation symbols.
# Need to loop over the repeating structure.
# Starting from the second payload block.
# The payload block length is 11.
# First we demodulate all the completely filled payload blocks.
current_payload_block_data_start_location = first_payload_block_start_location + (11*(L+N))
for i in range(0, total_number_of_complete_payload_blocks):
    current_payload_block_demodulated_data_symbols, current_payload_block_demodulated_pilot_symbols, current_payload_block_pilot_indices_overall = demodulate_pilot_tones(H_est_updated, current_payload_block_start_location, 10, received_signal, N, L, pilot_tones_start_index, pilot_tones_end_index, pilot_tones_step, first_frequency_bin, last_frequency_bin, update_sync_using_pilots=True)
    data_constellation_estimated = np.append(data_constellation_estimated, current_payload_block_demodulated_data_symbols)
    current_payload_block_data_start_location += (11*L+N)

# Now we demodulate any payload blocks that might be incompletely filled.
if (number_of_OFDM_symbols_in_incomplete_payload_block != 0):
    current_payload_block_demodulated_data_symbols, current_payload_block_demodulated_pilot_symbols, current_payload_block_pilot_indices_overall = demodulate_pilot_tones(H_est_updated, current_payload_block_start_location, number_of_OFDM_symbols_in_incomplete_payload_block, received_signal, N, L, pilot_tones_start_index, pilot_tones_end_index, pilot_tones_step, first_frequency_bin, last_frequency_bin, update_sync_using_pilots=True)
    data_constellation_estimated = np.append(data_constellation_estimated, current_payload_block_demodulated_data_symbols)



# Recover the output bits through LDPC decoding.
output_bits = LDPC_decoding(data_constellation_estimated, H_est_updated)
# TO-DO: Keep only the output bits that correspond to the actual transmitted data, and not the the other random symbols.
output_bits = np.array(output_bits[:file_size])

# TO-DO: In theory the line below should not be necessary.
#output_bits = np.array(output_bits)[:len(input_bits)]



# Compare output and input bits, and view the output.
input_bits = convert_file_to_bits('data_text.txt')
#print(bitlist_to_s(output_bits)) #uncommenting this should print the decoded data text
print(BER(input_bits, output_bits))
