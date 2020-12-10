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

class Viterbi:
    def __init__(self, msg, K=3):
        self.K = K # Constraint length
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
        # the first parity bit is the newest value xored with the second oldest value xored with the oldest bit
        # p0 = int(signal[i-2])^int(signal[i-1])^int(signal[i])
        # the second parity bit is the current value xored with the prior value
        # p1 = int(signal[i-1])^int(signal[i])
        return [str(0^int(s[0])^int(s[1]))+str(0^int(s[0])), # Parity bits if input is 0
                str(1^int(s[0])^int(s[1]))+str(1^int(s[0]))] # Parity bits if input is 1

    def _state_machine_gen(self,K):
        s = 0
        stm=np.zeros([2**K,2],dtype=object)
        for k in range(2**K):
            s=format(k,'#0'+str(K+2)+'b')[2:]
            stm[k]=self._parity_bits(s)
        return stm

    def _encode(self,signal):
        signal = "00" + signal
        transmit = ""
        x_n = 0 # initial value for x_n, the incoming message bit
        p_bts = "00" # initial value for parity bits string

        for i in range(2,len(signal)): #for every bit put into the encoder
            x_n = int(signal[i]) # the incoming message
            # The current state is string of {signal[i-1],signal[i-2]}
            # So take signal[i-2 to i-1] and flip it
            # Select out the correct column from the function output using the incoming message bit value
            p_bts = self._parity_bits(signal[i-2:i][::-1])[x_n]

            transmit = transmit + p_bts[x_n][0] + " " + p_bts[x_n][1] + " "
        return transmit

    def _despace_encode(self,signal):
        return ''.join(signal.split(" "))

    def re_encode(self,msg):
        self.msg = msg
        self.enc_sig_spaced = self._encode(self.msg)
        self.enc_sig = self._despace_encode(self.enc_sig_spaced)

    def decode(self,rxmsg,stmchn=self.state_machine):
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
        ### Initialize Trellis
        ns = len(stmchn) # num of trellis rows, which is num states bc state machine is list of lists
        r = len(stmchn[0][0]) # num of parity bits
        tl = int((len(rxmsg)/r)+1) # num of trellis columns, which is the number of parity bits is any given window
        trellisham = np.zeros([ns,tl],dtype=object) # trellis with ns rows, tl columns, and tuple values

        # Initialize first column in trellis
        trellisham[:,:]=np.inf
        trellisham[0,0]=(0,"nan")
        trellisham[1,0]=(np.inf,"nan")
        trellisham[2,0]=(np.inf,"nan")
        trellisham[3,0]=(np.inf,"nan")
        count = 0
        # print("initial trellis")
        # print(trellisham[:,:])

        ### filling the trellis
        for sym in range(0,len(rxmsg),r):
            count = count + 1 # what message portion are we on (start from 1)
            msg_section = rxmsg[sym:sym+r+1]
            for state in range(ns):
                s = bin(state)[2:].zfill(2) # extract the binary representation of the state
                p_as = s[1:] # predecessor state after shift, before adding in x[n]
                min_dp = (np.inf,0) # stores minimum hamming distance and predecessor state
                for p in [p_as+'1',p_as+'0']:
                    #print("pee",p)
                    p_dec = int(p,2) # decimal representation
                    x_n = s[0] # the value of x[n] that shifts from predecessor state to current state
                    ti = stmchn[p_dec][int(x_n)] # parity bits generated during transition
                    di = self.min_ham(msg_section, ti) # cost (hamming distance)
                    tot_di = trellisham[p_dec,count-1][0] + di
                    if di < min_dp[0]: # Check if cost is smaller than saved minimum
                        min_dp = (tot_di,p) # Update minimum cost and according state
                trellisham[state,count] = min_dp # Save minimum cost and according predecessor to trellis

        ### Backtracking

        # Find the total minimum cost of the most likely path

        min_cost = np.min(trellisham[:,-1])

        # Find the state associated with the most likely path
        end_state = np.argmin(trellisham[:,-1][0])
        #print("final trellis")
        #print(trellisham[:,:])
        #print(state_machine)
        # Backtrack and find the most likely sequence of states
        # Front-end insertion because backtracking (finds the last bit, first)
        # Use a deque because front-end insertion
        most_likely_states = collections.deque()
        # Use a second deque to keep track of predecssor states
        preds = collections.deque()
        preds.append(end_state) # add the state to start backtracking from
        # format(end_state,'#0'+str(len(stmchn[0,0])+2)+'b')[2:]
        # Begin backtracking
        states = []
        # print(end_state)
        states.append(format(end_state,'#0'+str(len(stmchn[0,0])+2)+'b')[2:])
        prev = trellisham[end_state,-1][1]
        states.append(prev)
        for bt_idx in np.flip(range(1,tl-1),0):
            prev = trellisham[int(prev,2),bt_idx][1]
            states.append(prev)
            #most_likely_states.appendleft(preds[-1])
            #print("bt_indx", bt_idx)
            #print("preds[-1]", preds[-1])
            #preds.append(trellisham[preds[-1],bt_idx])

        #most_likely_states = list(most_likely_states)
        # print("states",states)
        states.reverse()

        #return [b for b in most_likely_states[1:]]
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

if __name__ == "__main__":
    sig = generateRandomSignal(100)
    enc_sig = Encoder.Encoder(sig)
    enc_sig_2 = enc_sig.split(" ")
    enc_sig_3 = ''.join(enc_sig_2)
    noisy_sig = addSigNoise(enc_sig)
    K = 3 # Constraint length
    state_machine = Decoder.state_machine_gen(K-1)
    decoded_sig = Decoder.thedecoder_part2_tenyearslater(enc_sig_3,state_machine)

    print("Original",sig)
    print("Encoded",enc_sig)
    print("Noisy",noisy_sig)
    print("Decoded",decoded_sig)
    print("Hamming Distance between encoded and noisy signal", Decoder.min_ham(enc_sig_3, noisy_sig))
    print("Hamming Distance",Decoder.min_ham(decoded_sig,sig))

    mean_hamming_distance = 0
    for i in range(10000):
        sig = generateRandomSignal(100)
        enc_sig = Encoder.Encoder(sig)
        enc_sig_2 = enc_sig.split(" ")
        enc_sig_3 = ''.join(enc_sig_2)
        noisy_sig = addSigNoise(enc_sig)
        K = 3 # Constraint length
        state_machine = Decoder.state_machine_gen(K-1)
        decoded_sig = Decoder.thedecoder_part2_tenyearslater(noisy_sig,state_machine)
        mean_hamming_distance += Decoder.min_ham(decoded_sig,sig)
    mean_hamming_distance = mean_hamming_distance / 10000
    print("mean hamming", mean_hamming_distance)
