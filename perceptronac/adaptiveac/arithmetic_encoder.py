"""
Based on Mark Nelson's integer 16 bit implementation
from "The Data Compression Book".
"""


from perceptronac.adaptiveac.exceptions import EndOfBinaryFile


class ArithmeticEncoder:

    def __init__(self,inputFile, outputFile, nSymbols, symbolSize=8):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.nSymbols = nSymbols # should account for the termination symbol
        self.high = 0xFFFF
        self.low = 0
        self.underflowBits = 0
        self.symbolSize=symbolSize

    def __del__(self):
        self.inputFile.close()
        self.outputFile.close()

    def do_one_step(self,totals):
        # totals should account for the termination symbol
        try:
            c = self.inputFile.inputBits(self.symbolSize)
            self.update(c, totals)
            return c
        except EndOfBinaryFile:
            c = self.nSymbols-1
            self.update(c, totals) # termination symbol
            self.flush()
            self.outputFile.outputBits( 0, 16 ) # guarantees 16 bits for the decoder?
            return c

    def update(self, c, totals):
        
        scale = totals[self.nSymbols]
        lowCount = totals[ c ]
        highCount = totals[ c + 1 ]

        # .1111... is actually the same as 1  
        # 0xFFFF actually continues with Fs forever
        # 0xFFFF - 0x0000 is actually 0x10000 not 0xFFFF
        range = ( self.high-self.low ) + 1

        # Restrict the range of the number.
        # Before storing the new value of high, we need to decrement it
        self.high = self.low + (( range * highCount ) // scale - 1 ) 
        self.low = self.low + (( range * lowCount ) // scale )
        # print(f"symbol {c} low {self.low} high {self.high} lowCount {lowCount} highCount {highCount} scale {scale} range {range}.")

        # Prevent the range from getting too short
        while True:
            # low = 1... and high = 1... or low = 0... and high = 0...
            # If the MSBs became equal, they will not change anymore
            # in this case you can shift one place to the left.
            # Output the most significant bit. 
            # If there are any underflow bits, output them as well. 
            if ( ( self.high & 0x8000 ) == ( self.low & 0x8000 ) ) :
                # print(f"symbol {c} case 1 start low {self.low} high {self.high}")
                if ( self.high & 0x8000 ):
                    self.outputFile.outputBit( 1 )
                else:
                    self.outputFile.outputBit( 0 )
                while ( self.underflowBits > 0 ):
                    # example 1
                    #  
                    # assume low 0110 , high 1001
                    # high - low = 1001 - 0110  = 9 - 6  = 3 = 0011
                    # suppose highCount/scale=100%,lowCount/scale=100%
                    # low 1001, high 1001 
                    # you have to output 0 otherwise the tag will be higher than high
                    # 
                    # example 2
                    #  
                    # assume low 0110 , high 1001
                    # high - low = 1001 - 0110  = 9 - 6  = 3 = 0011
                    # suppose highCount/scale=0%,lowCount/scale=0
                    # low 0110, high 0110
                    # you have to output 1 otherwise the tag will be lower than low
                    #
                    if (~self.high & 0x8000):
                        self.outputFile.outputBit( 1 )
                    else:
                        self.outputFile.outputBit( 0 )
                    self.underflowBits = self.underflowBits-1

                self.low = self.low << 1
                self.high = self.high << 1
                self.high = self.high | 1
                self.high = self.high & 0xffff
                self.low = self.low & 0xffff
                # print(f"symbol {c} case 1 end low {self.low} high {self.high}")
            # Cannot have low = 1... and high = 0... , then low = 0... and high = 1...
            # If low = 01... and high = 10... the numbers are approaching each other
            # Remove the second MSB by making low = 00... and high = 01...
            # then shifting left by 1. The removed digit from high is always 0 and 
            # the removed digit from low is always 1. Keep track of how many digits 
            # you removed so you can put them back later.
            elif ( ( self.low & 0x4000 ) and not ( self.high & 0x4000 ) ):
                # print(f"symbol {c} case 2 start low {self.low} high {self.high}")
                self.underflowBits = self.underflowBits + 1
                self.low = self.low & 0x3fff # the two MSBs become 00
                self.high = self.high | 0x4000 # the two MSBs become 11
                
                self.low = self.low << 1
                self.high = self.high << 1
                self.high = self.high | 1
                self.high = self.high & 0xffff
                self.low = self.low & 0xffff
                # print(f"symbol {c} case 2 end low {self.low} high {self.high}")

            # Finish if
            # low = 00... and high = 10... or 
            # low = 00... and high = 11... or 
            # low = 01... and high = 11...
            else:
                # print(f"symbol {c} case 3 end low {self.low} high {self.high}")
                return

            # When shifting, you place a 0 in the low's least significant bit,
            # and a 1 in the high's least significant bit.
            # self.low = self.low << 1
            # self.high = self.high << 1
            # self.high = self.high | 1
            # self.high = self.high & 0xffff
            # self.low = self.low & 0xffff

    def flush(self):
        # low = 00... and high = 10... or 
        # low = 00... and high = 11... or 
        # low = 01... and high = 11...
        # we should continue outputting something between low and high
        # If low = 01... then it will output 10000...
        # If low = 00... then it will output 01111...
        if (self.low & 0x4000):
            self.outputFile.outputBit( 1 )
        else:
            self.outputFile.outputBit( 0 )
        self.underflowBits = self.underflowBits+1
        while ( self.underflowBits > 0 ):
            self.underflowBits = self.underflowBits - 1
            if ( ~self.low & 0x4000):
                self.outputFile.outputBit(1)
            else:
                self.outputFile.outputBit(0)

