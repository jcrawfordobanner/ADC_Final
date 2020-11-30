def Encoder(signal):
    signal = "00" + signal
    transmit = ""

    for i in range(2,len(signal)): #for every bit put into the encoder
        # the first parity bit is the newest value xored with the second oldest value xored with the oldest bit
        p0 = int(signal[i-2])^int(signal[i-1])^int(signal[i])
        # the second parity bit is the current value xored with the prior value
        p1 = int(signal[i-1])^int(signal[i])
        transmit = transmit+ str(p0)+str(p1) #encode parity bits to send to modulator

    return transmit

if __name__ == "__main__":
    test = Encoder("101100")
    check =  "111101000110"
    if(test == check):
        print("Hella")
    else:
        print(test)
        print("baka")

"""
    =^._.^=
    (=^･ｪ･^=))ﾉ彡☆
    /ᐠ｡▿｡ᐟ\*ᵖᵘʳʳ*
"""
