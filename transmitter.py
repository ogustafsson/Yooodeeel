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

N = 2048 # OFDM symbol length
block_size = N//2 - 1 # Set block size in frequency domain to half the OFDM symbol subtract the zeroed first frequency bin
L = 256 # Cyclic prefix length
n = 10 # Number of repetitions of known OFDM symbol for peforming channel estimation
sample_rate = 44100 # Sample rate is 44.1 kHz

# Define the QPSK constellation and its power
A = 10 # Radius of the QPSK constellation circle
bit_mapping = {(0,0): 0, (0,1): 1, (1,1): 2, (1,0): 3}# Use Gray coding to convert bit stream to constellation indices
constellation_mapping = A/np.sqrt(2) * np.array([(1+1j), (-1+1j), (-1-1j), (1-1j)]) # Define the mapping of indices to QPSK constellation in anti-clockwise sense
seed = 2021 # Standardise random number generator seed as 2021
rng = np.random.default_rng(seed) # Random number generator with specified seed

# Define the constants of the chirp signal for synchronisation; note that the duration is in seconds
chirp_constants = {"startFrequency": 100, "endFrequency": 20000, "duration": 0.1, "method": "linear"}

FILE_NAME_TRANSMIT = "audio_to_transmit.wav"

c = ldpc.code()

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

# TODO: figure out these functions, where to put them?
def generate_random_indices(block_size):
    """Generates numpy array of random indices with seed '2021' such that random array is predictable"""
    random_constellation_indices = rng.integers(low=0, high=4, size=block_size) # Generate N/2 - 1 random indices for the QPSK constellation

    return random_constellation_indices

# Generate an array of 511 random constellation symbols with '0' as the first element.
def generate_random_sequence(block_size):
    """Takes the size of block (N/2-1) with information, assuming QPSK encoding and returns array of random constellation symbols with length (N/2-1)"""
    #block_size = int(N / 2 - 1) # Set size of block to OFDM symbol length divided by 2 for mirroring and subtract 1 to have zeroed first entry

    random_constellation_indices = generate_random_indices(block_size) # Generate random indices of block size length

    random_sequence = map_indices_to_constellation(random_constellation_indices)

    return random_sequence # Length is N/2 - 1 !!!BE CAREFUL, THIS HAD A ZERO FIRST FREQUENCY BIN BEFORE!!!

def create_header(filename):
    """
    Takes a file associated with the filename and generates a header to capture the metadata.
    This includes file length and file type with protection added for safety.
    """
    pass
    return header

def convert_file_to_bits(filename):
    """Takes a file associated with the filename and converts it first to bytes, then to a bit stream"""
    # file is txt file
    with open(filename, "r") as text:
        text_string = text.read()
    return s_to_bitlist(text_string)
def encode_bits(source_bits):
    tem = []
    for i in range(int(len(source_bits)//c.K)):
        tem.extend(c.encode(source_bits[i*c.K:(i+1)*c.K]))
    return np.array(list(tem))

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

def split_sequence_into_blocks(constellation_sequence):
    """
    Takes a QPSK constellation sequence and splits it into (N/2 - 1) blocks
    For the last block, we pad the end with random constellation symbols
    """
    constellation_sequence_length = len(constellation_sequence) # Find the overall length of the constellation sequence
    number_of_blocks_excluding_last = constellation_sequence_length // block_size # Calculate the number of individual blocks, NOT including the last block

    remainder = constellation_sequence_length % block_size # Calculate the remainder: number of frequency bins left over in the last block
    sequence_excluding_remainder = constellation_sequence[:-remainder] # Slice sequence to exclude constellation symbols in the last block

    blocks_excluding_last = np.split(sequence_excluding_remainder, number_of_blocks_excluding_last) # Split sequence into blocks of length (N/2 - 1)

    # If we have the rare case where there is no remainder, we are done
    if remainder == 0:
        blocks = blocks_excluding_last
    # For most cases where there is a remainder, we need to pad with random constellation symbols
    else:
        random_constellation_symbols_length = block_size - remainder # Find the length of the constellation symbols to add on
        # NOTE: this next step may change depending on the standard
        random_constellation_symbols = generate_random_sequence(random_constellation_symbols_length) # Generate a random sequence of constellation symbols

        last_block = np.append(constellation_sequence[-remainder:], random_constellation_symbols) # Create the last block by appending the random symbols to the remaining sequence

        # UNIT TEST: check last block is correct length
        assert len(last_block) == block_size, "Last block is not (N/2-1) long"

        blocks = np.vstack((blocks_excluding_last, last_block)) # Add generated last block to array of blocks

    # NOTE: we return the original sequence length so that we can discard the padding in the receiver
    return blocks, constellation_sequence_length # Return array of (N/2-1) long blocks as well as the length of the constellation sequnce for later use

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

def generate_data_signal(data_blocks):
    """Takes a set of blocks of QPSK constellation symbols and generates a data signal in time-domain"""
    data_signal = np.array([]) # Set up empty array
    # Generate OFDM symbols from each block and concatenate into long data signal
    for block in data_blocks:
        OFDM_symbol = generate_OFDM_symbol(block)
        data_signal = np.append(data_signal, OFDM_symbol)

    return data_signal

def generate_repeated_signal(OFDM_TX_withCP):
    """Generate signal for performing channel estimation by repeating the OFDM symbol n times"""
    repeated_signal = np.array([]) # Set up empty array
    # Repeats OFDM symbol n times
    for i in range(0,n):
        repeated_signal = np.append(repeated_signal, OFDM_TX_withCP) # TODO: find a more efficient method

    # UNIT TEST: check length of repeated signal
    assert len(repeated_signal) == n*(N+L), "Repeated signal should be n*(N+L) long, but is not"
    return repeated_signal

# Overall master function for generation estimation signal
def generate_estimation_signal(n):
    """Generates a signal of n-repeated known pseudo-random OFDM symbols"""
    random_block = generate_random_sequence(block_size) # 1. Generate random block of constellation symbols in frequency-domain
    OFDM_symbol = generate_OFDM_symbol(random_block) # 2. Generate OFDM symbol in time-domain
    estimation_signal = generate_repeated_signal(OFDM_symbol) # 3. Generate estimation signal

    return estimation_signal

def set_up_overall_signal(chirp_signal, estimation_signal, data_signal):
    """Set up overall signal to be transmitted by concatenating the three components"""
    overall_signal = np.concatenate((chirp_signal, estimation_signal, data_signal))
    return overall_signal

def generate_signal_for_transmission(data_file):
    """
    Master function for generating the signal for transmission over an audio channel
    Input: name of the file to be transmitted
    Output: writes audio .wav file to disk
    """

    # Prepare data signal
    bit_stream = convert_file_to_bits(data_file) # 1. Convert data file to bit stream
    encoded_bit_stream = encode_bits(bit_stream) # 2. Encode bit stream with LPDC codes
    #encoded_bit_stream = rng.integers(low=0, high=2, size=int(block_size*10.5)) # For basic testing

    constellation_indices = map_bits_to_constellation_indices(encoded_bit_stream) # 3.1 Map encoded bit stream to constellation indices
    constellation_sequence = map_indices_to_constellation(constellation_indices) # 3.2 Map constellation indices to QPSK constellation symbols
    data_blocks, data_constellation_length = split_sequence_into_blocks(constellation_sequence) # 4. Split sequence of constellation symbols into blocks and add random padding
    data_signal = generate_data_signal(data_blocks) # 6. Generate data signal by converting frequency-domain blocks to time-domain OFDM symbols

    # Prepare estimation signal
    estimation_signal = generate_estimation_signal(n) # 7. Generate signal for performing estimation using repeated known OFDM symbols

    # Prepare chirp signal
    chirp_signal = generateChirp(chirp_constants, sample_rate) # 8. Generate chirp signal for performing initial synchronisation

    # Prepare signal for transmission
    overall_signal = set_up_overall_signal(chirp_signal, estimation_signal, data_signal) # 9. Set up overall signal with three components
    convert_array_to_wav(overall_signal, sample_rate, FILE_NAME_TRANSMIT) # 10. Write to disk audio .wav file for transmisison over channel
