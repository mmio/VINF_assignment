import io
import csv
import gzip
import itertools

import multiprocessing

def get_text_from_gzip(archives):
    # unzip multiple files
    text_streams = map(
        lambda archive:
            io.TextIOWrapper(
                gzip.GzipFile(fileobj=archive, mode='r')
            ),
        archives)

    # combine iterators
    text_stream = itertools.chain(*text_streams)

    # cnt = 0
    # for row in csv.reader(text_stream, delimiter="\t"):
    #     yield row
    #     cnt += 1
    #     if cnt == 1_000:
    #         break

    return csv.reader(text_stream, delimiter="\t")

def get_texts_from_gzip(archives):
    text_streams = map(
        lambda archive:
            io.TextIOWrapper(
                gzip.GzipFile(fileobj=archive, mode='r')
            ),
        archives)

    return map(lambda ts: csv.reader(ts, delimiter="\t"), text_streams)
