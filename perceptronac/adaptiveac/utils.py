
from exceptions import EndOfBinaryFile
from exceptions import CorruptedFile
import math

# part of the ascii table for displaying purposes (see asciitable.com) 
ascii_table = {
    0: "NUL", 1: "SOH", 2: "STX", 3: "ETX",
    4: "EOT", 5: "ENQ", 6: "ACK", 7: "BEL",
    8: "BS", 9: "HT", 10: "LF", 11: "VT",
    12: "FF", 13: "CR", 14: "SO", 15: "SI",
    16: "DLE", 17: "DC1", 18: "DC2", 19: "DC3",
    20: "DC4", 21: "NAK", 22: "SYN", 23: "ETB",
    24: "CAN", 25: "EM", 26: "SUB", 27: "ESC",
    28: "FS", 29: "GS", 30: "RS", 31: "US",
    127: "del"
}


def printCounts(counts):
    print("\n")
    print( "symbol\tchar\tcount")
    for i in range(len(counts)):
        if (counts[i] != 0 and i <= 255 ):
            print(f"{i}\t({ascii_table[i] if ascii_table.get(i) else chr(i)})\t{counts[i]}")
        elif (counts[i] != 0 and i > 255 ):
            print(f"{i}\t\t{counts[i]}")
    print("\n")


def sourceEntropy(counts):
    a = 0
    probs = []
    for i in range(len(counts)):
        if (counts[i] > 0):
            probs.append(counts[i])
            a = a + counts[i]
    for i in range(len(probs)):
        probs[i] = probs[i] / a
    
    probs = list(map(lambda p: -p*math.log2(p),probs))
    entropy = sum(probs)
    return entropy


def printEntropy(counts):
    print("\n")
    print(sourceEntropy(counts))
    print("\n")


def limitCountsSumTo14bits(counts):
    """
    The sum of all elements of 'counts' must be 2 bits smaller than the 'low'
    and 'high' variables of the arithmetic coder, which are 16 bits, and 2^14 = 16384.
    """

    a = sum(counts)
    scale = 16000.0/a 
    # If you multiply all terms of the sum by this, the sum will be 16000.
    # Not scale = 16383/a because we are limiting all terms below by 1.
    # 383 is the amount of free space we are giving to make this possible
    # and still keep the sum below 16383.
    if(a>16383):
        counts = list(map(lambda v: 0 if v == 0 else int(max(1,v*scale)),counts))
    a = sum(counts)
    assert(a < 16384)
    return counts


def getCounts(inputFile):
    longCounts = []
    charCounts = []
    for _ in range(256):
        longCounts.append(0)
        charCounts.append(0)

    while True:
        try:
            c = inputFile.inputBits(8)
        except EndOfBinaryFile:
            break
        longCounts[ c ] = longCounts[ c ] + 1

    longMax = max(longCounts)

    for i in range(256):
        charCounts[i] = int(255*((1.0 * longCounts[i]) / longMax))
        if (charCounts[i] == 0 and longCounts[i] !=0):
            charCounts[i] = 1

    inputFile.reset()

    return charCounts


def readCounts(inputFile):
    charCounts=[]
    try:
        nSymbols = inputFile.inputBits(16)
    except EndOfBinaryFile:
        raise CorruptedFile
    for _ in range(nSymbols):
        try:
            c = inputFile.inputBits(8)
        except EndOfBinaryFile:
            raise CorruptedFile
        charCounts.append(c)
    return charCounts

def compressionRatio(inputFile, outputFile):
    numerator = inputFile.size()
    denominator = outputFile.size()
    r = ((1.0 * numerator) / denominator)
    return r

def avgLength(inputFile, outputFile):
    numerator = outputFile.size()
    denominator = inputFile.size()/8 + 1
    return (numerator / denominator)

def defineIntervals(counts):
    nSymbols = len(counts)
    totals = []
    # The ranges are obtained by cummulative counts.
    # For symbol x, the lower bound can be found at totals[ x ], 
    # the upper bound at totals[ x + 1 ].
    # Therefore we need to keep track of nSymbols+1 numbers.
    for _ in range(nSymbols+1):
        totals.append(0)
    for i in range(nSymbols):
        totals[i+1] = totals[i] + counts[i]
    return nSymbols,totals 

def printIntervals(nSymbols,totals):
    print("symbol\tupper bound")
    for i in range(nSymbols-1,-1,-1):
        print(f"{i}\t{totals[i+1]}")

def static_model(inputFile):
    counts = getCounts(inputFile) # 256 symbols
    counts.append(1) # 257 symbols. Count for the termination symbol = 1 (minimum count)
    counts = limitCountsSumTo14bits(counts); # sum of all counts should be 2 bits less than 16
    nSymbols,totals = defineIntervals(counts)
    return counts,nSymbols,totals
