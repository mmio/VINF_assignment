import os
import sys

import lucene
from java.nio.file import Paths
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField, StringField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version

from tqdm import tqdm

if __name__ == "__main__":
  lucene.initVM()

  subfolder_index_pairs = [
    ('cluster_w2v', 'index_w2v'),
    ('cluster_w2v_n', 'index_w2v_n'),
    ('cluster_tfidf', 'index_tfidf'),
    ('cluster_tfidf_n', 'index_tfidf_n')
  ]

  for subfolder, index in subfolder_index_pairs:
    print(f'Indexing {subfolder} with {index}')

    path = Paths.get(index)
    indexDir = SimpleFSDirectory(path)

    writerConfig = IndexWriterConfig(StandardAnalyzer())
    writer = IndexWriter(indexDir, writerConfig)

    for day in tqdm(os.listdir('data/dates/')):
      path = f'data/dates/{day}/{subfolder}/'
      if os.path.exists(path):
        for cluster_id in os.listdir(path):
          with open(f'{path}{cluster_id}', 'r') as cluster_file:
            text = cluster_file.read()

            doc = Document()
            doc.add(Field("cluster_id", f'{day}-{cluster_id}', TextField.TYPE_STORED))
            doc.add(Field("content", text, TextField.TYPE_STORED))
            writer.addDocument(doc)
    writer.close()
