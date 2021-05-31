# Auxilliary function to the main function for computing the timing offset
def compute_autocorrelation_sum(signal, position):
    """Takes signal in the time-domain and evalutes the autocorrelation function at specified position over n-repeated symbols"""
    symbol_length = N + L # Define OFDM symbol (including cyclic prefix) length for clarity
    
    # Sum for n-1 times, where n is the number of repeated symbols.
    # For example, if there are 10 repeated symbols, this sum should take autocorrelation between symbols 1 and 2, 2 and 3, ..., 9 and 10 which is 9 iterations
    sum = 0
    for i in range(n-1):
        sum += signal[position+i*symbol_length] * signal[position+(i+1)*symbol_length] # TODO: Do we take the complex conjugate for the first term?

    return sum

# Main function for computing timing offset of start of known OFDM symbols
def compute_timing_offset(signal):
    """
    Takes a time-domain signal and computes the timing offset of the start of the known OFDM symbols.
    This function uses a modified Schmidl and Cox method, which computes the maximum of a metric which peaks at the start of the repeated symbols.
    """

    symbol_length = N + L # Define OFDM symbol (including cyclic prefix) length for clarity

    d_set = np.arange(len(signal)) # Create a set of values at which to evaluate the synchronisation metric P
    padded_signal = np.append(signal, np.zeros(n*symbol_length)) # Pad with n*(N+l) zeroes to avoid errors in computing autocorrelation function

    P = np.zeros(len(d_set)) # Create empty array to put metric values in later
    # Initialise synchronsiation metric at zero position by taking autocorrelation sum (over each repeated symbol) over the whole symbol length N+L
    P[0] = np.sum(np.array([compute_autocorrelation_sum(padded_signal, m) for m in range(symbol_length)]))

    # Compute synchronsiation metric P at all positions, except 0, by recursion
    for d in d_set[:-1]:
        P[d+1] = P[d] + compute_autocorrelation_sum(padded_signal, d+symbol_length) - compute_autocorrelation_sum(padded_signal, d)

    P = np.abs(P) # Take the absolute value of P
    M = P / np.max(P) # Normalise synchronisation metric by the maximum value

    timing_offset = np.argmax(M) # Compute timing offset by simple argmax

    return timing_offset