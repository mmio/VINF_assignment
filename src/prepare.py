import io
import csv
import gzip
import itertools

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

    # text_stream = io.TextIOWrapper(
    #     gzip.GzipFile(fileobj=archives[0], mode='r'),
    #     newline=''
    # )

    # read as tsv
    return csv.reader(text_stream, delimiter="\t")