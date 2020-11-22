import os
import sys
import lucene
 
from java.nio.file import Paths
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version

if __name__ == "__main__":
  lucene.initVM()
  
  path = Paths.get("index/")
  if sys.argv[1] == 'norm':
    path = Paths.get("index_norm/")
    
  indexDir = SimpleFSDirectory(path)

  writerConfig = IndexWriterConfig(StandardAnalyzer())
  writer = IndexWriter(indexDir, writerConfig)

  cluster = 'clusters'
  if sys.argv[1] == 'norm':
    cluster = 'clusters_norm'
  
  for folder in os.listdir('data/dates/'):
    path = f'data/dates/{folder}/{cluster}/'
    if os.path.exists(path):
      print(f'indexing {folder}')
      for sub_folder in os.listdir(path):
        with open(f'{path}{sub_folder}', 'r') as cluster_file:
          print(f'cluster {sub_folder}')
          text = cluster_file.read()

          doc = Document()
          doc.add(Field("cluster_id", f'{folder}_{sub_folder}', TextField.TYPE_STORED))
          doc.add(Field("content", text, TextField.TYPE_STORED))
          writer.addDocument(doc) 

#  with open('data/dates/5_25/queries', 'r') as f:
#    doc = Document()
#    doc.add(Field("text", f, TextField.TYPE_STORED))
#    writer.addDocument(doc)

  print('Closing writer')
  writer.close()
