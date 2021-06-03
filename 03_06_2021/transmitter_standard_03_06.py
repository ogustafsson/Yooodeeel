# Import numpy as main library for array manipulations
import numpy as np

# Import signal processing functions from scipy library (chirp, correlation)
from scipy import signal

# Import data manipulation libraries
import pandas as pd
import csv

# Import matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Import the fast Fourier transform and its inverse from the scipy library
from scipy import fft
from scipy import ifft

# Import library for manipulating data formats
import bitarray

# Import functions for audio manipulation
from scipy.io.wavfile import write # For writing from a numpy array to a .wav file
from scipy.io.wavfile import read # For reading from a .wav file to an numpy array

import ldpc
import math
N = 2048 # OFDM symbol length
block_size = N//2 - 1 # Set block size in frequency domain to half the OFDM symbol subtract the zeroed first frequency bin
L = 256 # Cyclic prefix length
n = 10 # Number of repetitions of known OFDM symbol for peforming channel estimation
sample_rate = 44100 # Sample rate is 44.1 kHz
a = 50 #included
b = 700 #excluded
# Define the QPSK constellation and its power
A = np.sqrt(2) # Radius of the QPSK constellation circle
bit_mapping = {(0,0): 0, (0,1): 1, (1,1): 2, (1,0): 3}# Use Gray coding to convert bit stream to constellation indices
mapping_table = {(0,0): (1+1j), (0,1): (-1+1j), (1,1): (-1-1j), (1,0): (1-1j)}
constellation_mapping = A/np.sqrt(2) * np.array([(1+1j), (-1+1j), (-1-1j), (1-1j)]) # Define the mapping of indices to QPSK constellation in anti-clockwise sense
seed_2020 = 2020
seed_2021 = 2021
seed_2022 = 2022
rng_2020 = np.random.default_rng(seed_2020) # Random number generator with specified seed
rng_2021 = np.random.default_rng(seed_2021)
rng_2022 = np.random.default_rng(seed_2022)
rng = np.random.default_rng(seed_2022)
# Define the constants of the chirp signal for synchronisation; note that the duration is in seconds
chirp_constants = {"startFrequency": 100, "endFrequency": 10000, "duration": 1, "method": "logarithmic"}
pilot_value = 1+1j
FILE_NAME_TRANSMIT = "audio_to_transmit.wav"

c = ldpc.code(z = 81)
k = c.K #data bits need to be a multiple of k in order to do LDPC encoding/Decoding

# Useful helper functions for converting between numpy arrays and audio .wav files.
def convert_array_to_wav(array, sample_rate, filename):
    """Takes a signal in a numpy array and the sampling rate to generate an audio .wav file with file name 'output.wav'"""
    write(filename, sample_rate, array)

def convert_wav_to_array(received_filename):
    """Takes the name of the audio .wav file and returns the conversion into a numpy array"""
    rate, array = read(received_filename)

    return array

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


# Functions performing the DFT and iDFT.
def DFT(signal):
    """Compute the discrete Fourier transform of a signal"""
    return fft(signal)

def iDFT(signal):
    """Compute the inverse discrete Fourier transform of a signal"""
    return ifft(signal)

# Returns on the real part of the iDFT.
def iDFT_real(signal):
    """"Compute the inverse discrete Fourier transform of a signal and take the real part"""
    modulated_signal = ifft(signal) # Take the inverse discrete Fourier transform
    # TODO: perform a check that the imaginary component is effectively zero
    modulated_signal_real = modulated_signal.real # Take the real component of the transformed signal
    return modulated_signal_real

# Generate chirp signal
def generateChirp(chirp_constants, sample_rate):
    """Generates a chirp signal with dictionary of chirp constants including: start and end frequency, duration in seconds and the method of frequency sweep"""
    number_of_samples = int(sample_rate*chirp_constants["duration"]) # Calculate the number of samples of the chirp by multiplying the duration in seconds by the sampling rate
    t_samples = np.array(range(0, number_of_samples)) # Create an array with equally spaced sample indices
    t_seconds = t_samples / sample_rate # Divide by the sampling frequency to generate the time indices in seconds

    chirp_signal = signal.chirp(t_seconds, chirp_constants["startFrequency"], chirp_constants["duration"], chirp_constants["endFrequency"], method=chirp_constants["method"]) # Use scipy library to generate chirp signal

    return chirp_signal

#update this
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
# Note: for the chirp, this index will be located halfway along the chirp.
def matched_filter(noisy_signal, desired_signal):
    """Matched filter returns the index of the highest correlation between two signals"""
    desired_signal = desired_signal.flatten() # Flatten the numpy array to ensure no problems
    correlation = signal.correlate(noisy_signal, desired_signal, mode='same') # Compute the correlation between the two signals

    max_correlation = np.amax(correlation) # Calculate the max value of the correlation
    max_correlation_index = np.argmax(correlation) # Calculate the index of that maximum value

    return max_correlation_index

# Find index of the highest correlation of a chirp signal with itself
def find_chirp_start_index(chirp_signal):
    """Uses a matched filter to find index of the highest correlation of a chirp signal with itself"""
    chirp_length = len(chirp_signal)
    chirp_start_index = matched_filter(chirp_signal, chirp_signal)
    assert chirp_start_index == int(chirp_length/2), "Chirp start index does not coincide with half of chirp length"
    return chirp_start_index


def generate_random_bit_sequence(bit_sequence_length, random_seed):
    #random seed is the number corresponding to the random seed, aka 2020, 2021, 2022, etc
    rng = np.random.default_rng(random_seed)
    random_bit_sequence = rng.integers(low=0, high=2, size = bit_sequence_length)
    return random_bit_sequence

def random_bit_sequence_to_constellation_symbols(random_bit_sequence):
    return np.array([mapping_table[tuple(b)] for b in bits])

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

# Note: The two concatenated arrays both begin with a '0' in the DC frequency bin.
def generate_symmetric_sequence(block):
    """Takes a (N/2-1) block, appends the reversed complex-conjugate form, and sets DC bins to zero"""
    block_with_zero = np.append(np.array([0]), block) # Set the first frequency bin to zero
    reversed_block_with_zero = np.append(np.array([0]), np.conj(np.flip(block))) # Reverse the order of the input array, and take the complex conjugate.

    symmetric_sequence = np.append(block_with_zero, reversed_block_with_zero) # Append the flipped array to the original array.

    # UNIT TEST: check length of output symmetric sequence
    assert len(symmetric_sequence) == N, "Length of symmetric sequence should be N but is not"

    return symmetric_sequence

# Note: inverse discrete Fourier transform has been defined in global functions as iDFT_real()

def add_cyclic_prefix(OFDM_TX_noCP):
    """Takes time-domain OFDM symbol without cyclic prefix and inserts cyclic prefix of length L"""
    cyclic_prefix = OFDM_TX_noCP[-L:] # Generate cyclic prefix by taking last L values
    OFDM_TX_withCP = np.append(cyclic_prefix, OFDM_TX_noCP) # Create single OFDM symbol by joining cyclic prefix and sequence

    # UNIT TEST: check length of OFDM symbol
    assert len(OFDM_TX_withCP) == N+L, "Single OFDM symbol should be length N+L but is not" # Check length of single OFDM symbol as N+L

    return OFDM_TX_withCP

# Note: this function acts on one block at a time
def generate_OFDM_symbol(block):
    """
    Input: (N/2-1)-long block of QPSK constellation symbols in frequency-domain
    Output: N-long OFDM symbol (with cyclic prefix) in time-domain
    """

    # UNIT TEST: check length of block input to this function
    assert len(block) == block_size, "Block is not N/2-1 long"

    symmetric_sequence = generate_symmetric_sequence(block) # 1. Generate a symmetric sequence with zeroed DC frequency bins
    OFDM_TX_noCP = iDFT_real(symmetric_sequence) # 2. Modulate symmetric sequence into time-domain by taking the iDFT
    OFDM_TX_withCP = add_cyclic_prefix(OFDM_TX_noCP) # 3. Add a cyclic prefix of length L

    # UNIT TEST: check length of OFDM symbol
    assert len(OFDM_TX_withCP) == N+L, "Output OFDM symbol with cyclic prefix is not N+L long"

    return OFDM_TX_withCP


def create_random_OFDM_signal_time(number_of_blocks, random_seed):
    """
    Input: number of blocks of length N = 2048 we want to generate all at once (number_of_blocks = 1 for seed_2020, number_of_blocks = 5 for seed_2021, number_of_blocks = 1 for seed_2022)
    Output: number_of_blocks*(N+L) time domain long sequence
    """
    bit_sequence_length = number_of_blocks*(N-2)
    random_bit_sequence = generate_random_bit_sequence(bit_sequence_length, random_seed)
    constellation_indices = map_bits_to_constellation_indices(random_bit_sequence)
    blocks = map_indices_to_constellation(constellation_indices)
    block_size = N//2 - 1
    blocks = np.split(blocks, len(blocks)/block_size)

    output_sequence = np.array([])
    for block in blocks:
        output_sequence = np.append(output_sequence, generate_OFDM_symbol(block))
    return output_sequence

#Functions for converting Image to Bit_Array
def access_bit(data, num):
    base = int(num // 8)
    shift = int(num % 8)
    return (data[base] & (1<<shift)) >> shift

def byte_array_to_bit_array(byte_array):
    return [access_bit(byte_array,i) for i in range(len(byte_array)*8)]

def image_to_bitarray(filename):
    with open(filename, "rb") as image:
      f = image.read()
      image_data = bytearray(f)
    return byte_array_to_bit_array(image_data)



def convert_file_to_bits(filename):
    extension = filename[-4:]
    if (extension == '.txt'):
        with open(filename, "r") as text:
            text_string = text.read()
        data_bits = s_to_bitlist(text_string)
    if (extension == 'tiff'):
        data_bits = image_to_bitarray(filename)
    if (extension == '.wav'):
        pass
    return data_bits
def create_header(filename):
    """
    Takes a file associated with the filename and generates a header to capture the metadata.
    This includes file length and file type with protection added for safety.
    Header is a
    """
    extension = filename[-4:]
    extension_dictionary = {
    'tiff': [0,0,1,1,0,0,1,0],
    '.txt': [0,1,0,0,1,0,0,0],
    '.wav': [1,0,1,0,0,1,0,1]
    }
    file_type = extension_dictionary[extension]
    file_type_repeated = np.tile(file_type, 5)

    file_to_bits = convert_file_to_bits(filename)
    file_size = len(file_to_bits)

    string_bits = bin(file_size)[2:]

    bits_size = [int(c) for c in string_bits]

    zero_padding = 32 - len(bits_size)
    bits_size = np.append(np.zeros(zero_padding), bits_size)
    file_size_repeated = np.tile(bits_size, 5)

    header = np.concatenate([file_type_repeated, file_size_repeated])
    assert len(header) == 200, "Header is not created properly its size is not 200"
    return header

def convert_file_to_bits_include_header(filename):
    #we assume that the file is .txt
    """Takes a file associated with the filename and converts it first to bytes, then to a bit stream"""
    # file is txt file
    extension = filename[-4:]
    if (extension == '.txt'):
        with open(filename, "r") as text:
            text_string = text.read()
        data_bits = s_to_bitlist(text_string)
    if (extension == 'tiff'):
        data_bits = image_to_bitarray(filename)
    if (extension == '.wav'):
        pass
    header_bits = create_header(filename)

    return np.concatenate([header_bits, data_bits])

def encode_bits(source_bits):
    source_bits_length = len(source_bits)
    additional_bits = 0
    if (source_bits_length % k != 0):
        additional_bits = k - source_bits_length % k

    #Generate random source bits to make the data bits length a multiple of k in order to enable proper LDPC encoding
    rng = np.random.default_rng(2023)
    random_bits = rng.integers(0, 2, additional_bits)

    source_bits = np.append(source_bits, random_bits)

    tem = []
    for i in range(int(len(source_bits)//c.K)):
        tem.extend(c.encode(source_bits[i*c.K:(i+1)*c.K]))
    return np.array(list(tem))

def generate_random_indices(block_size):
    """Generates numpy array of random indices with seed '2021' such that random array is predictable"""
    rng = np.random.default_rng(2025)
    random_constellation_indices = rng.integers(low=0, high=4, size=block_size) # Generate N/2 - 1 random indices for the QPSK constellation

    return random_constellation_indices

# Create OFDM symbols and add pilot tones to each symbol.
def generate_random_sequence(block_size):
    """Takes the size of block (N/2-1) with information, assuming QPSK encoding and returns array of random constellation symbols with length (N/2-1)"""
    #block_size = int(N / 2 - 1) # Set size of block to OFDM symbol length divided by 2 for mirroring and subtract 1 to have zeroed first entry

    random_constellation_indices = generate_random_indices(block_size) # Generate random indices of block size length

    random_sequence = map_indices_to_constellation(random_constellation_indices)

    return random_sequence # Length is N/2 - 1 !!!BE CAREFUL, THIS HAD A ZERO FIRST FREQUENCY BIN BEFORE!!!

def split_sequence_into_blocks_with_pilot_tones(constellation_sequence):

  pilot_indices = np.arange(1, 1018, 8)
  random_data_indices = np.array([])
  data_indices = np.array([])

  all_indices = np.arange(N//2)

  for index in all_indices:
      if (not np.isin(index, pilot_indices)):
          if (index >= b or index < a):
              random_data_indices = np.append(random_data_indices, index)
          else:
              data_indices = np.append(data_indices, index)

  number_of_pilot_tones = len(pilot_indices)
  number_of_data_tones = len(data_indices)
  number_of_random_tones = len(random_data_indices)

  rng = np.random.default_rng(2023)
  random_bits = rng.integers(0, 2, 2*number_of_random_tones)
  random_tones_indices = map_bits_to_constellation_indices(random_bits)
  random_tones = map_indices_to_constellation(random_tones_indices)

  constellation_sequence_length = len(constellation_sequence) # Find the overall length of the constellation sequence
  number_of_blocks_excluding_last = constellation_sequence_length // number_of_data_tones # Calculate the number of individual blocks, NOT including the last block

  remainder = constellation_sequence_length % number_of_data_tones # Calculate the remainder: number of frequency bins left over in the last block
  sequence_excluding_remainder = constellation_sequence[:-remainder] # Slice sequence to exclude constellation symbols in the last block

  blocks_excluding_last = np.split(sequence_excluding_remainder, number_of_blocks_excluding_last) # Split sequence into blocks of length (N/2 - 1)

  blocks = np.array([])
  count = 0
  for block in blocks_excluding_last:
    random_index = 0
    current_block = np.array([])
    current_index = 0
    for i in range(N//2-1):
      if (np.isin(i, pilot_indices)):
          current_block = np.append(current_block, pilot_value)
      if (np.isin(i, data_indices)):
          current_block = np.append(current_block, block[current_index])
          current_index += 1
      if (np.isin(i, random_data_indices)):
          current_block = np.append(current_block, random_tones[random_index])
          random_index += 1

    if count == 0:
      blocks = current_block
      count += 1
    else:
      blocks = np.vstack((blocks, current_block))

  if remainder != 0:
    #this is quite a random step as in I created way more random_constellation_symbols than neccessary, needs to be fixed :)
    random_constellation_symbols_length = number_of_data_tones - remainder # Find the length of the constellation symbols to add on
    # NOTE: this next step may change depending on the standard
    random_constellation_symbols = generate_random_sequence(10*random_constellation_symbols_length) # Generate a random sequence of constellation symbols

    last_block = np.append(constellation_sequence[-remainder:], random_constellation_symbols) # Create the last block by appending the random symbols to the remaining sequence

    #this needs to be fixed
    current_block = np.array([])
    current_index = 0
    random_index = 0
    for i in range(N//2-1):
      if (np.isin(i, pilot_indices)):
        current_block = np.append(current_block, pilot_value)
      if (np.isin(i, data_indices)):
        current_block = np.append(current_block, last_block[current_index])
        current_index += 1
      if (np.isin(i, random_data_indices)):
        current_block = np.append(current_block, random_tones[random_index])
        random_index += 1

    blocks = np.vstack((blocks, current_block))

  return blocks, constellation_sequence_length # Return array of (N/2-1) long blocks as well as the length of the constellation sequnce for later use


def generate_data_signal_with_preamble_blocks(data_blocks):
    """Takes a set of blocks of QPSK constellation symbols and generates a data signal in time-domain"""
    number_of_data_blocks = len(data_blocks)
    preamble_count = number_of_data_blocks // 10

    if (number_of_data_blocks % 10 != 0):
        preamble_count += 1

    preamble_signal = generate_preamble_signal(2022)

    all_indices = np.arange(preamble_count*11)
    preamble_indices = list(range(0, preamble_count*11, 11))
    preamble_indices = np.array(preamble_indices)
    data_indices = np.delete(all_indices, preamble_indices)

    data_signal = np.array([]) # Set up empty array
    # Generate OFDM symbols from each block and concatenate into long data signal
    block_index = 0
    for index in all_indices:
        if np.isin(index, preamble_indices):
            data_signal = np.append(data_signal, preamble_signal)
        else:
            if (block_index < len(data_blocks)):
                OFDM_symbol = generate_OFDM_symbol(data_blocks[block_index])
                block_index = block_index + 1
                data_signal = np.append(data_signal, OFDM_symbol)
    return data_signal


def generate_repeated_signal(OFDM_TX_withCP, number_of_repetitions):
    """Generate signal for performing channel estimation by repeating the OFDM symbol n times"""
    repeated_signal = np.array([]) # Set up empty array
    # Repeats OFDM symbol n times
    for i in range(0,number_of_repetitions):
        repeated_signal = np.append(repeated_signal, OFDM_TX_withCP) # TODO: find a more efficient method

    # UNIT TEST: check length of repeated signal
    #assert len(repeated_signal) % (N+L) == 0, "Repeated signal should be number_of_repetitions*(N+L) long, but is not"]
    return repeated_signal

# Overall master function for generation estimation signal
def generate_estimation_signal(random_seed):
    """Generates a signal of n-repeated known pseudo-random OFDM symbols"""
    estimation_signal = create_random_OFDM_signal_time(5, random_seed)
    estimation_signal = generate_repeated_signal(estimation_signal, 2)

    return estimation_signal

def generate_preamble_signal(random_seed):
    """Generates preamble by first generating a single random OFDM block using random_seed, and converting it to time domain"""
    preamble_signal = create_random_OFDM_signal_time(1, random_seed)
    return preamble_signal

def map_indices_to_constellation(constellation_indices):
    """Takes sequence of constellation indices from range {0,1,2,3} and maps them to a sequence of QPSK constellation symbols"""

    # UNIT TEST: check constellation indices belong to range {0,1,2,3}
    assert not np.isin([False], np.isin(constellation_indices, [0,1,2,3])), "Indices do not belong to range {0,1,2,3}"

    constellation_sequence = np.array([]) # Set up empty array for the loop
    # Map each random index to its corresponding constellation symbol
    for index in constellation_indices:
        constellation_sequence = np.append(constellation_sequence, constellation_mapping[index]) # TODO: make this loop more efficient

    return constellation_sequence
def generate_data_signal(data_file):
    """
    Master function for generating the signal for transmission over an audio channel
    Input: name of the file to be transmitted
    Output: writes audio .wav file to disk
    """
    # Prepare data signal
    bit_stream = convert_file_to_bits_include_header(data_file) # 1. Convert data file to bit stream
    encoded_bit_stream = encode_bits(bit_stream) # 2. Encode bit stream with LPDC codes
    #encoded_bit_stream = rng.integers(low=0, high=2, size=int(block_size*10.5)) # For basic testing

    constellation_indices = map_bits_to_constellation_indices(encoded_bit_stream) # 3.1 Map encoded bit stream to constellation indices
    constellation_sequence = map_indices_to_constellation(constellation_indices) # 3.2 Map constellation indices to QPSK constellation symbols
    data_blocks, data_constellation_length = split_sequence_into_blocks_with_pilot_tones(constellation_sequence) # 4. Split sequence of constellation symbols into blocks and add random padding
    data_signal = generate_data_signal_with_preamble_blocks(data_blocks) # 6. Generate data signal by converting frequency-domain blocks to time-domain OFDM symbols

    return data_signal


def generate_signal_for_transmission(data_file):
    """
    Master function for generating the signal for transmission over an audio channel
    Input: name of the file to be transmitted
    Output: writes audio .wav file to disk
    """
    chirp_signal = generate_chirp()
    preamble_signal = generate_preamble_signal(2020)
    estimation_signal = generate_estimation_signal(2021)
    data_signal = generate_data_signal(data_file)

    overall_signal = np.concatenate((chirp_signal, preamble_signal, estimation_signal, data_signal, chirp_signal))

    convert_array_to_wav(overall_signal, sample_rate, FILE_NAME_TRANSMIT)

    return overall_signal

overall_signal = generate_signal_for_transmission("tiff_image.tiff") #this data length is used as a global constant in receiver.py
