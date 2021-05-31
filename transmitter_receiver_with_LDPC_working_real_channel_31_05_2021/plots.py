#Plot the received signal.
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


# Plot the estimated frequency response and impulse response of the channel.
# Magnitude Frequency Response.
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
