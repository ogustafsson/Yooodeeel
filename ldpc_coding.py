import ldpc
import numpy as np
from bitarray import bitarray
from PIL import Image
import binascii


#Defining constants and mapping/demapping tables
mapping_table = {
    (0,0) : 1+1j,
    (0,1) : -1+1j,
    (1,1) : -1-1j,
    (1,0): 1-1j
}
demapping_table = {v : k for k, v in mapping_table.items()}
noise_variance = 1


# Defining all the neccessary functions
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


c = ldpc.code()


FILE_NAME = "hamlet.txt"
with open(FILE_NAME, "r") as text:
    hamlet_text = text.read()

hamlet_text = hamlet_text[:3000]
source_bits = s_to_bitlist(hamlet_text)
source_bits = source_bits[:c.K*10]
#source_bits = rand_bin_array(c.K*4,c.K*32)
#source_bits = np.array(list(source_bits))
input_bits = source_bits
#
#Encoder

encoded_source_bits = encode_source_bits(source_bits)
bits_SP = SP(encoded_source_bits)

modulated_bits = mapping(bits_SP)
noise_real = np.random.normal(0,noise_variance,len(modulated_bits))
noise_imag = np.random.normal(0,noise_variance,len(modulated_bits))

modulated_bits.real = modulated_bits.real + noise_real
modulated_bits.imag = modulated_bits.imag + noise_imag


output = modulated_bits #constellation symbols with added noise
channel_frequency = np.ones(len(output))



# Decoder
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


input_bits = np.array(input_bits)
output_bits = np.array(output_bits)
#print(bitlist_to_s(output_bits))
print(BER(input_bits, output_bits))
