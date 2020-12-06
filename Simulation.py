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
    signal = 1*(np.random.random(desired_length)>0.5)
    return ''.join([str(b) for b in signal])

def addNoise(encodedSignal):
    """
    takes in encoded signal and adds noise to it
    """
    new_sig = encodedSignal.split(" ")
    print(new_sig[0:-1])
    for i in range(len(new_sig)-1):
        new_sig[i] = str(round(np.random.normal(0,1)/2+int(new_sig[i]))%2)
    return "".join(new_sig)

if __name__ == "__main__":
    sig = generateRandomSignal(100)
    enc_sig = Encoder.Encoder(sig)
    enc_sig_2 = enc_sig.split(" ")
    enc_sig_3 = ''.join(enc_sig_2)
    noisy_sig = addNoise(enc_sig)
    K = 3; # Constraint length
    state_machine = Decoder.state_machine_gen(K-1);
    decoded_sig = Decoder.thedecoder_part2_tenyearslater(enc_sig_3,state_machine)
    print("Original",sig)
    print("Encoded",enc_sig)
    print("Noisy",noisy_sig)
    print("Decoded",decoded_sig)
    print("Hamming Distance",Decoder.min_ham(decoded_sig,sig))
