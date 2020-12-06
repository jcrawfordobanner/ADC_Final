import numpy as np
import collections

def min_ham(rx,trial):
    dist = 0;
    for i in range(len(trial)):
        if(trial[i] != rx[i]):
            dist = dist + 1;
    return dist

def parity_bits(s):
    return [str(0^int(s[0])^int(s[1]))+str(0^int(s[0])), # Parity bits if input is 0
            str(1^int(s[0])^int(s[1]))+str(1^int(s[0]))] # Parity bits if input is 1

def state_machine_gen(K):
    s = 0
    stm=np.zeros([2**K,2],dtype=object)
    for k in range(2**K):
        s=format(k,'#0'+str(K+2)+'b')[2:]
        stm[k]=parity_bits(s)
    return stm

def thedecoder_part2_tenyearslater(rxmsg,stmchn):
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
                di = min_ham(msg_section, ti) # cost (hamming distance)
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

if __name__ == "__main__":
    print("Testing decoding.")
    test =  "111101000110"
    check = "101100"
    K = 3; # Constraint length
    state_machine = state_machine_gen(K-1);
    decoded = thedecoder_part2_tenyearslater(test,state_machine)
    print("decoded",decoded)
    if(decoded==check):
        print("YAY")
    else:
        print("BOO")
