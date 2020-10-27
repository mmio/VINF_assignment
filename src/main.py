# https://github.com/explosion/spaCy/blob/01aec7a313753775603a9e6f752f75cc16ac43fb/examples/pipeline/multi_processing.py#L48

import contextlib

import json
from functools import partial
from collections import defaultdict

from spacy.util import minibatch

from files import urls, destination
from download import download_and_save
from prepare import get_text_from_gzip
from pipeline import get_pipe

from processors.histogramOfTokens import HistogramOfTokens
from processors.histogramOfQueries import HistogramOfQueries

def main():
    # open multiple files
    archives = [
        download_and_save(url, destination)
        for url in urls
    ]

    # Collect global stats
    processors = [
        # HistogramOfTokens(),
        # HistogramOfNormalized(),
        HistogramOfQueries('output/hoq.json'),
        # DictionaryOfTokens(),
        # DictionaryOfNormalized()
    ]

    # nlp = get_pipe()
    # query_stream = map(lambda i: i[1], get_text_from_gzip(archives))
    # for doc in nlp.pipe(query_stream):
    #     for processor in processors:
    #         processor.add_doc(doc)

    # for p in processors:
    #     p.save()

    d = defaultdict(list)
    for row in get_text_from_gzip(archives):
        try:
            d[row[0]].append(row[1])
        except KeyError:
            d[row[0]] = [row[1]]
    for k in d:
        if len(d[k]) > 300:
            print(k, len(d[k]))

    # fp = open('output/userQueries.json', 'w')
    # json.dump(d, fp)
    # fp.close()

    # d = defaultdict(list)
    # for row in get_text_from_gzip(archives):
    #     try:
    #         d[row[2]].append(row[1])
    #     except KeyError:
    #         d[row[2]] = [row[1]]

    # fp = open('output/timeQueries.json', 'w')
    # json.dump(d, fp)
    # fp.close()

    # d = defaultdict(list)
    # for row in get_text_from_gzip(archives):
    #     try:
    #         d[row[0]].append((row[1], row[2]))
    #     except KeyError:
    #         d[row[0]] = [(row[1], row[2])]

    # fp = open('output/userQueries.json', 'r')
    # json.dump(d, fp)
    # fp.close()

    # avg = []
    # counter = 0
    # with open('output/userQueries.json') as f:
    #     data = json.load(f)
    #     for item in data:
    #         print(len(data[item]))


        # data = json.load(f)
        # for doc in nlp.pipe(map(lambda r: r[0], data['479'])):
        #     if len(avg) == 0:
        #         avg = doc.vector
        #     else:
        #         avg += doc.vector
        #     counter += 1
        # print(avg/counter)

if __name__ == '__main__':
    main()