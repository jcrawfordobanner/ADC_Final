"""
In order to test the encoder and decoder together, we'll simulate a random signal being encoded and then decoded

The main steps are:
-generate a random signal
-encode it
-add noise to it
-decode it

"""
import Decoder
import Encoder
import numpy as np

def generateRandomSignal(desired_length):
    """
    Create random binary signal with input length
    """
    #signal = np.random.randbytes(desired_length) # this only works with random 3.9
    signal = np.random.random()
    return(signal)

def addNoise(encodedSignal):
    """
    takes in encoded signal and adds noise to it
    """
    noise = np.random.normal(0,1,len(encodedSignal))
    noise = np.where(noise > 0.5, 1, 0)

def main():
    sig = generateRandomSignal(100)
    enc_sig = Encoder(sig)
    noisy_sig = addNoise(enc_sig)
