import numpy as np

def decoder(rxmsg,stmchn):
    """
    rxmsg - received message
    stmchn - state machine:
        array:
            rows = all possible states
            columns = message bit value (0 or 1)
            value = string of parity bits
    """
    ns = len(stmchn) # num of trellis rows, which is num states bc state machine is list of lists
    r = len(stmchn[0][0]) # num of parity bits
    tl = (len(rxmsg)/r)+1 # num of trellis columns, which is the number of parity bits is any given window
    trellis = np.array([ns,tl,2],dtype=int) # trellis with ns rows, tl columns, and tuple values
    # Initialize first column in trellis
    trellis[:,0,0]=np.inf
    trellis[0,0,0]=0
    # Iterate and update the rest of the trellis
    # for sym in rxmsg[::len(stmchn[0][0])]: # only gives the first element, not the whole section
    count = 0
    for sym in range(0,len(rxmsg),r)
        count = count + 1 # what message portion are we on (start from 1)
        msg_section = rxmsg(sym:sym+r+1)
        for state in range(ns):
            s = bin(state)[2:] # extract the binary representation of the state
            p_as = s[1:] # predecessor state after shift, before adding in x[n]
            min_dp = [0,0] # stores minimum hamming distance and predecessor state
            for p in [p_as+'1',p_as+'0']:
                p_dec = int(p,2) # decimal representation
                x_n = s[0] # the value of x[n] that shifts from predecessor state to current state
                ti = stmchn[p_dec][x_n] # parity bits generated during transition
                di = min_ham(msg_section, ti) # hamming distance
                if di < min_dp[0]:
                    min_dp = [di,p]
            trellis[state][count] = min_dp



def min_ham(rx,trial):
    dist = 0;
    for i in range(len(trial)):
        if(trial[i] ~= rx[i]):
            dist++;
    return dist

"""
State machine 2d = row state column state value tuple (transmitted bit, received bit)
Trellis values = 2d array (row is state. column is bit, value is hamming distance for most likely  bit)
Trellis prior states = 2d array (row is state. column is bit, value is most likely prior state )

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
"""
