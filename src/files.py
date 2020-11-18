destination = 'data/datasets/aol'

base = 'http://www.cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/'

files = [
    'user-ct-test-collection-02.txt.gz',
    # 'user-ct-test-collection-03.txt.gz',
    # 'user-ct-test-collection-04.txt.gz',
    # 'user-ct-test-collection-05.txt.gz',
    # 'user-ct-test-collection-06.txt.gz',
    # 'user-ct-test-collection-07.txt.gz',
    # 'user-ct-test-collection-08.txt.gz',
    # 'user-ct-test-collection-09.txt.gz',
    # 'user-ct-test-collection-10.txt.gz'
]

urls = [
    f'{base}{file}'
    for file in files
]