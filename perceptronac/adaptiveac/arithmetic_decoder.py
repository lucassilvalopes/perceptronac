"""
Based on Mark Nelson's integer 16 bit implementation
from "The Data Compression Book".
"""


from perceptronac.adaptiveac.exceptions import CorruptedFile, EndOfBinaryFile


class ArithmeticDecoder:

    def __init__(self,inputFile, outputFile,nSymbols,symbolSize=8):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.nSymbols = nSymbols
        self.high = 0xFFFF
        self.low = 0
        self.underflowBits = 0
        self.code = None
        self.symbolSize=symbolSize

    def __del__(self):
        self.inputFile.close()
        self.outputFile.close()

    def do_one_step(self,totals):
        try:
            if self.code is None:
                # read the first 16 bits of the tag
                self.code = self.inputFile.inputBits(16)
            c = self.getSymbol(totals)
            if ( c == self.nSymbols-1 ):
                return 1
            else:
                self.update(c, totals)
                self.outputFile.outputBits(c, self.symbolSize)
                return 0
        except EndOfBinaryFile:
            raise CorruptedFile

    def getSymbol(self, totals):
        # This operation is only needed to find which symbol's portion 
        # of the interval "code" falls in. "code" is left unaltered.
        scale = totals[self.nSymbols]
        range = ( self.high-self.low ) + 1
        # ((code - low) / range) * scale , but rearranged
        count = ( ( self.code - self.low + 1 ) * scale-1 ) // range 
        
        c = self.nSymbols-1
        while (count < totals[ c ]):
            c = c - 1 
        return c

    def update(self, c, totals):
        
        scale = totals[self.nSymbols]
        lowCount = totals[ c ]
        highCount = totals[ c + 1 ]

        range = ( self.high-self.low ) + 1
        self.high = self.low + (( range * highCount ) // scale - 1 )
        self.low = self.low + (( range * lowCount ) // scale )
        # print(f"symbol {c} low {self.low} high {self.high} lowCount {lowCount} highCount {highCount} scale {scale} range {range}.")
        while True:
            # If the MSBs are equal, just shift left by 1 bit 
            # (you do not need to keep track of the underflow digits)
            if ( ( self.high & 0x8000 ) == ( self.low & 0x8000 ) ):
                # print(f"symbol {c} case 1 start low {self.low} high {self.high}")
                self.low = self.low << 1
                self.high = self.high << 1
                self.high = self.high | 1
                self.code = self.code << 1
                self.code = self.code + self.inputFile.inputBit()

                self.high = self.high & 0xffff
                self.low = self.low & 0xffff
                self.code = self.code & 0xffff
                # print(f"symbol {c} case 1 end low {self.low} high {self.high}")
            # This "if" condition is the same as before. The only
            # difference is that you also remove the next-MSB from "code". 
            # And you do that by toggling the next-MSB from "code"
            # and then shifting it left by 1 bit. 
            # Note that "code" is either 01... or 10... 
            # because it is between low and high which are 01... and 10...
            # When you toggle it, it becomes 00... or 11 respectively.
            elif ( ( self.low & 0x4000 ) and not ( self.high & 0x4000 ) ):
                # print(f"symbol {c} case 2 start low {self.low} high {self.high}")
                self.code = self.code ^ 0x4000
                self.low = self.low & 0x3fff
                self.high = self.high | 0x4000

                self.low = self.low << 1
                self.high = self.high << 1
                self.high = self.high | 1
                self.code = self.code << 1
                self.code = self.code + self.inputFile.inputBit()

                self.high = self.high & 0xffff
                self.low = self.low & 0xffff
                self.code = self.code & 0xffff
                # print(f"symbol {c} case 2 end low {self.low} high {self.high}")
            else: 
                # print(f"symbol {c} case 3 end low {self.low} high {self.high}")
                return

            # self.low = self.low << 1
            # self.high = self.high << 1
            # self.high = self.high | 1
            # self.code = self.code << 1
            # self.code = self.code + self.inputFile.inputBit()

            # self.high = self.high & 0xffff
            # self.low = self.low & 0xffff
            # self.code = self.code & 0xffff

