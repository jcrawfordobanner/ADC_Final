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
import matplotlib.pyplot as plt
import random
from os import path
import csv

class Viterbi:
    def __init__(self, msg, K=5, r=4):
        self.K = K # Constraint length
        self.r = r # Number of parity bits
        self.state_machine = self._state_machine_gen(self.K-1) # k-1 is the state window length
        self.msg = msg
        self.enc_sig_spaced = self._encode(self.msg)
        self.enc_sig = self._despace_encode(self.enc_sig_spaced)

    def min_ham(self,rx,trial):
        dist = 0;
        for i in range(len(trial)):
            if(trial[i] != rx[i]):
                dist = dist + 1;
        return dist

    def _parity_bits(self,s):
        """
        Using generator polynomials: (11101) and (11010)
        Accordingly assumes that the state provided is of window size 4
        """
        # the first parity bit is the newest value xored with the second oldest value xored with the oldest bit
        # the second parity bit is the current value xored with the prior value

        # Make it possible to loop through conveniently using K and r
        if self.K == 3:
            p0 = int(s[0])^int(s[1])
            p1 = int(s[0])
            if self.r == 2:
                return [str(0^p0)+str(0^p1), # Parity bits if input is 0
                        str(1^p0)+str(1^p1)] # Parity bits if input is 1
            else:
                p2 = int(s[1])
                return [str(0^p0)+str(0^p1)+str(0^p2)+str(0), # Parity bits if input is 0
                        str(1^p0)+str(1^p1)+str(1^p2)+str(1)] # Parity bits if input is 1
        elif self.K == 5:
            p0 = int(s[0])^int(s[1])^int(s[3])
            p1 = int(s[0])^int(s[2])
            if self.r == 2:
                return [str(0^p0)+str(0^p1), # Parity bits if input is 0
                        str(1^p0)+str(1^p1)] # Parity bits if input is 1
            else:
                p2 = int(s[0])^int(s[1])^int(s[2])
                p3 = int(s[0])^int(s[1])^int(s[2])^int(s[3])
                return [str(0^p0)+str(0^p1)+str(0^p2)+str(0^p3), # Parity bits if input is 0
                        str(1^p0)+str(1^p1)+str(1^p2)+str(0^p3)] # Parity bits if input is 1
        else:
            p0 = int(s[0])^int(s[1])
            p1 = int(s[0])
            return [str(0^p0)+str(0^p1), # Parity bits if input is 0
                    str(1^p0)+str(1^p1)]

    def _state_machine_gen(self,K):
        s = 0
        stm=np.zeros([2**K,2],dtype=object)
        for k in range(2**K):
            s=format(k,'#0'+str(K+2)+'b')[2:]
            stm[k]=self._parity_bits(s)
        return stm

    def _encode(self,signal):
        signal = "".zfill(self.K-1) + signal
        transmit = ""
        x_n = 0 # initial value for x_n, the incoming message bit
        p_bts = "".zfill(len(self.state_machine[0,0])) # initial value for parity bits string

        for i in range(self.K-1,len(signal)): #for every bit put into the encoder
            x_n = int(signal[i]) # the incoming message
            # The current state is string of {signal[i-1],signal[i-2]}
            # So take signal[i-2 to i-1] and flip it
            # Select out the correct column from the function output using the incoming message bit value
            p_bts = self._parity_bits(signal[i-self.K+1:i][::-1])[x_n]

            transmit = transmit + " ".join(p_bts) + " "
        return transmit

    def _despace_encode(self,signal):
        return ''.join(signal.split(" "))

    def re_encode(self,msg):
        self.msg = msg
        self.enc_sig_spaced = self._encode(self.msg)
        self.enc_sig = self._despace_encode(self.enc_sig_spaced)

    def decode(self,rxmsg,stmchn=None):
        """
        State machine 2d = row state column state value tuple (transmitted bit, received bit)
        Trellis values = 2d array
                row is state
                column is bit
                value is a tuple of hamming distance for most likely  bit, and most likely prior state


        Given that you’ve received L bits, and that each message bit encodes R parity bits:

        From 1 -> L/R (iterate variable called i):
        ri = received message bits for i-th segment
        For all states si:
            For all predecessor states pi:
                ti = parity bits for transition.
                di = Hamming distance between ri and ti
            dmin = min(all di)
            pmin = min(predecessor state that got dmin)
            Store in Trellis values dmin for position (si,i)
            Store in Trellis prior states pmin for position (si,i)

        At the very end, find the send where dmin is the smallest. That’s the ending state.
        From there, trace backwards what pmin was to the very beginning.

        rxmsg - received message
        stmchn - state machine:
            array:
                rows = all possible states
                columns = message bit value (0 or 1)
                value = string of parity bits
        """
        ### Initialize state machine
        stmchn = self.state_machine if stmchn==None else stmchn
        ### Initialize Trellis
        ns = len(stmchn) # num of trellis rows, which is num states bc state machine is list of lists
        r = len(stmchn[0][0]) # num of parity bits
        tl = int((len(rxmsg)/r)+1) # num of trellis columns, which is the number of parity bits is any given window
        trellisham = np.zeros([ns,tl],dtype=object) # trellis with ns rows, tl columns, and tuple values

        # Initialize first column in trellis
        trellisham[:,:]=np.inf
        trellisham[:,0]=[(np.inf,"nan")]
        trellisham[0,0]=(0,"nan")
        # trellisham[1,0]=(np.inf,"nan")
        # trellisham[2,0]=(np.inf,"nan")
        # trellisham[3,0]=(np.inf,"nan")
        count = 0
        # print("initial trellis")
        # print(trellisham[:,:])

        ### filling the trellis
        for sym in range(0,len(rxmsg),r):
            count = count + 1 # what message portion are we on (start from 1)
            msg_section = rxmsg[sym:sym+r+1]
            for state in range(ns):
                s = bin(state)[2:].zfill(self.K-1) # extract the binary representation of the state
                p_as = s[1:] # predecessor state after shift, before adding in x[n]
                min_dp = (np.inf,0) # stores minimum hamming distance and predecessor state
                for p in [p_as+'1',p_as+'0']:
                    p_dec = int(p,2) # decimal representation
                    x_n = s[0] # the value of x[n] that shifts from predecessor state to current state
                    # print("p",p,"pdec",p_dec,"x",x_n,"s",s,"c",count)
                    ti = stmchn[p_dec][int(x_n)] # parity bits generated during transition
                    di = self.min_ham(msg_section, ti) # cost (hamming distance)
                    tot_di = trellisham[p_dec,count-1][0] + di
                    # print("tot",trellisham[p_dec,count-1][0])
                    if tot_di < min_dp[0]: # Check if cost is smaller than saved minimum
                        min_dp = (tot_di,p) # Update minimum cost and according state
                trellisham[state,count] = min_dp # Save minimum cost and according predecessor to trellis

        ### Backtracking

        # Find the total minimum cost of the most likely path

        min_cost = np.min(trellisham[:,-1])

        # Find the state associated with the most likely path
        end_state = np.argmin(trellisham[:,-1])
        #print("final trellis")
        #print(trellisham[:,:])
        #print(state_machine)
        # Backtrack and find the most likely sequence of states
        # Begin backtracking
        states = []
        # print(end_state)
        states.append(format(end_state,'#0'+str(self.K-1+2)+'b')[2:])
        prev = trellisham[end_state,-1][1]
        states.append(prev)
        for bt_idx in np.flip(range(1,tl-1),0):
            prev = trellisham[int(prev,2),bt_idx][1]
            states.append(prev)

        #most_likely_states = list(most_likely_states)
        # print("states",states)
        states.reverse()

        #return [b for b in most_likely_states[1:]]
        #print(trellisham)
        return ''.join([b[0] for b in states[1:]])

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

def addSigNoise(encodedSignal):
    """
    takes in encoded signal and adds noise to it
    """
    new_sig = encodedSignal.split(" ")
    #print(new_sig)
    for i in range(len(new_sig)-1):
        if(new_sig[i]=="1"):
            new_sig[i]=5
        elif(new_sig[i]=="0"):
            new_sig[i]=-5
        new_sig[i] = round(np.random.normal(0,5)+new_sig[i])
        if(new_sig[i]>0):
            new_sig[i] = "1"
        elif(new_sig[i]<=0):
            new_sig[i]="0"
    #print(new_sig)
    return "".join(new_sig)

def addDistributedNoise(encodedSignal, lenNoise):
    """
    takes in encoded signal and adds noise to it by adding noise to random bits in the signal

    parameter lenNoise is an integer that represents the number of bits that will be exposed to noise
    """
    new_sig = encodedSignal.split(" ") #splits signal into list of bits
    random_bits  = random.sample(range(len(new_sig)-1),lenNoise) #takes a random sample of indices from the list, the random sample has length lenNoise

    for i in random_bits: #for each random
        new_sig[i] = str(round(np.random.normal(0,1)+int(new_sig[i]))%2)
    return "".join(new_sig)

def save_to_csv( signal_length, constraint_length, noise_length,
                    avg_hamming_distance, filename="simulation_results.csv", clearing_file=False):
        """
        Same simulation metrics to csv
        """

        metrics = {
            "length_signal" : signal_length,
            "constraint_length" : constraint_length,
            "noise_length" : noise_length,
            "avg_hamming_distance": avg_hamming_distance,

        }


        new_file = not path.exists(filename)

        options = 'w+' if clearing_file else 'a'  # Truncate the file if clearing_file set True, else append to file

        with open(filename, options) as csvfile:
            csvwriter = csv.writer(csvfile)

            if new_file or clearing_file:
                csvwriter.writerow(metrics.keys())

            csvwriter.writerow(metrics.values())
def generate_metrics():
    kr_list = ([3,2], [3,4],[5,2], [5,4])

    for z in kr_list:
        for j in range(100,600,50):
            for q in np.linspace(0,1,10):

                signal_length = j
                K = z[0]
                r = z[1]
                noise_length = int(signal_length*q)
                mean_hamming_distance = 0



                for i in range(100):
                    #sig = generateRandomSignal(signal_length)
                    #vit.re_encode(sig)
                    sig = generateRandomSignal(signal_length)
                    vit = Viterbi(sig, K=K, r=r)
                    vit.re_encode(sig)
                    #print("sig", sig)
                    #noisy_sig = addSigNoise(enc_sig)

                    noisy_sig = addDistributedNoise(vit.enc_sig_spaced, noise_length)
                    decoded_sig = vit.decode(noisy_sig)
                    mean_hamming_distance += vit.min_ham(decoded_sig,sig)
                mean_hamming_distance = mean_hamming_distance / 100
                save_to_csv(signal_length = signal_length, constraint_length = z, noise_length = noise_length,
                                    avg_hamming_distance = mean_hamming_distance, filename="simulation_results.csv", clearing_file=False)
                print("k_r",z,"signal length", signal_length, "mean hamming", mean_hamming_distance)


if __name__ == "__main__":

    # sig = generateRandomSignal(1000)
    # #sig = "110000"
    # vit = Viterbi(sig,3,2)
    # #noisy_sig = addSigNoise(vit.enc_sig_spaced)
    # decoded_sig = vit.decode(vit.enc_sig)
    # print("Original",sig)
    # print("Encoded",vit.enc_sig_spaced)
    # print("Decoded ",decoded_sig)
    # print("Hamming Distance",vit.min_ham(decoded_sig,sig))
    #generate_metrics()

    kr_list = ([3,2], [3,4],[5,2], [5,4])

    for z in kr_list:
        for j in range(100,600,50):


            signal_length = j
            K = z[0]
            r = z[1]

            mean_hamming_distance = 0
            for i in range(100):
                #sig = generateRandomSignal(signal_length)
                #vit.re_encode(sig)
                sig = generateRandomSignal(signal_length)
                vit = Viterbi(sig, K=K, r=r)
                vit.re_encode(sig)
                #print("sig", sig)
                noisy_sig = addSigNoise(vit.enc_sig_spaced)
                decoded_sig = vit.decode(noisy_sig)
                mean_hamming_distance += vit.min_ham(decoded_sig,sig)
            mean_hamming_distance = mean_hamming_distance / 100
            save_to_csv(signal_length = signal_length, constraint_length = z, noise_length = "NA",
                                avg_hamming_distance = mean_hamming_distance, filename="simulation_results_awgn.csv", clearing_file=False)
            print("k_r",z,"signal length", signal_length, "mean hamming", mean_hamming_distance)
