class MultipleFiles():
    def __init__(self, filelimit, mode = 'r+'):
        self.mode = mode
        self.limit = filelimit

    def __enter__(self):
        self.cleancounter()
        return self

    def __exit__(self, type, value, traceback):
        self.closeall()

    def open(self, filename):
        if (fd := self.openfiles.get(filename, False)):
            return fd
        
        self.closeonlimit()
        return self.addfile(filename)

    def addfile(self, filename):
        fp = open(filename, self.mode)
        self.openfiles.update({filename: fp})
        self.filescount += 1
        print(self.filescount)
        return fp

    def closeonlimit(self):
        if self.filescount >= self.limit:
            print("closing on limit")
            self.closeall()

    def closeall(self):
        for value in self.openfiles.values():
            value.close()
        self.cleancounter()

    def cleancounter(self):
        self.openfiles = {}
        self.filescount = 0

# with MultipleFiles(100, 'r') as mf:
#     for i in range(200):
#         mf.open('MultipleFileHandler.py')