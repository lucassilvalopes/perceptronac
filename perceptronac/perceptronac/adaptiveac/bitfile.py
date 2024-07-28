"""
https://www.devdungeon.com/content/working-binary-data-python
https://stackoverflow.com/questions/2872381/
https://stackoverflow.com/questions/67480215/
https://stackoverflow.com/questions/5131647/
https://codippa.com/how-to-check-file-size-in-python-3-ways-to-find-out-size-of-file-in-python/
"""

from perceptronac.adaptiveac.exceptions import EndOfBinaryFile, WriteError

class BitFile:

    def __init__(self,fn,fs):
        """
        Args:
            fn : file name
            fs : flags "rb" , "wb", "ab"
        """
        self.flags = fs
        self.buff = 0x00 # 0000 0000
        self.mask = 0x80 # 1000 0000
        self.stream = open(fn, self.flags)


    def outputBit(self,bit):
        if ( bit != 0 ):
            self.buff = self.buff | self.mask
        self.mask = self.mask >> 1
        if ( self.mask == 0 ):
            num_bytes_written = self.stream.write(self.buff.to_bytes(1, byteorder='big'))
            if(not num_bytes_written):
                raise WriteError
            self.buff = 0x00
            self.mask = 0x80


    def outputBits(self,code,bitCount):
        longMask = 0x01 << ( bitCount - 1 )
        while ( longMask != 0):
            if ( longMask & code ):
                self.buff = self.buff | self.mask
            self.mask = self.mask >> 1
            longMask = longMask >> 1
            if ( self.mask == 0 ):
                num_bytes_written = self.stream.write(self.buff.to_bytes(1, byteorder='big'))
                if(not num_bytes_written):
                    raise WriteError
                self.buff = 0x00
                self.mask = 0x80
            

    def inputBit(self):
        if (self.mask == 0x80):
            b = self.stream.read(1)
            if(not b):
                raise EndOfBinaryFile
            self.buff = int.from_bytes(b, byteorder='big')
        
        value = self.buff & self.mask
        self.mask = self.mask >> 1
        if ( self.mask == 0 ):
            self.mask = 0x80
        return (1 if value else 0)


    def inputBits(self, bitCount):
        longMask = 0x01 << ( bitCount - 1 )
        code = 0x00
        while ( longMask != 0):
            if ( self.mask == 0x80 ):
                b = self.stream.read(1)
                if(not b):
                    raise EndOfBinaryFile
                self.buff = int.from_bytes(b, byteorder='big')

            if ( self.buff & self.mask ):
                code = code | longMask
            
            longMask = longMask >> 1
            self.mask = self.mask >> 1
            if ( self.mask == 0 ):
                self.mask = 0x80

        return code


    def close(self):
        if (self.mask != 0x80):
            if (self.flags == "wb"):
                num_bytes_written = self.stream.write(self.buff.to_bytes(1, byteorder='big'))
                if(not num_bytes_written):
                    raise WriteError
        self.stream.close()


    def reset(self):
        self.stream.seek(0)


    def size(self):
        """
        Size in bytes. 
        Use after you have input or output all bytes but not yet closed the file
        """
        sz = self.stream.tell()
        if (self.mask != 0x80):
            if (self.flags == "wb"):
                sz=sz+1; # if there is still one byte to be written when closing
        return sz
