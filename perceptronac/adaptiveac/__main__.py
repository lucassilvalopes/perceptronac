
import sys
from bitfile import BitFile
from arithmetic_encoder import ArithmeticEncoder
from arithmetic_decoder import ArithmeticDecoder
from utils import compressionRatio, static_model, printCounts, printEntropy, \
    printIntervals, avgLength

def main():

    if(len(sys.argv) < 4):
        print("example of use: ")
        print(f"{sys.argv[0]} encoderinput encoderoutput decoderoutput")

    encoderInputFileName = sys.argv[1]
    encoderOutputFileName = sys.argv[2]
    decoderOutputFileName = sys.argv[3]

    encoderInputFile = BitFile(encoderInputFileName, "rb")
    encoderOutputFile = BitFile(encoderOutputFileName, "wb")

    counts,nSymbols,totals = static_model(encoderInputFile)

    print("Scaled Counts: ")
    printCounts(counts)
    print("Intervals: ")
    printIntervals(nSymbols,totals)
    print("Entropy: ")
    printEntropy(counts)

    enc = ArithmeticEncoder(encoderInputFile, encoderOutputFile, nSymbols)
    
    done = 0
    while not done:
        done = enc.do_one_step(totals)

    print("Average Length: ")
    print("\n")
    print(f"\n{avgLength(encoderInputFile, encoderOutputFile)}\n")
    print("Compression Ratio: ")
    print(f"\n{compressionRatio(encoderInputFile, encoderOutputFile)}\n")
    
    del encoderInputFile
    del encoderOutputFile
    del enc

    decoderInputFile = BitFile(encoderOutputFileName, "rb")
    decoderOutputFile = BitFile(decoderOutputFileName, "wb")
    dec = ArithmeticDecoder(decoderInputFile, decoderOutputFile, nSymbols)

    done = 0
    while not done:
        done = dec.do_one_step(totals)


if __name__ == "__main__": 
    main()