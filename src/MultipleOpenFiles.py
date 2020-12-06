class MultipleOpenFiles():
    def __init__(self):
        self.openFiles = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for openFile in self.openFiles.values():
            openFile.close()

    def writeline(self, fileId, text):
        self.openFiles.get(fileId).write(f'{text}\n')

    def get(self, fileId):
        return self.openFiles.get(fileId, False)

    def add(self, fileId, destination):
        self.openFiles.update({fileId: open(f'{destination}', 'a')})