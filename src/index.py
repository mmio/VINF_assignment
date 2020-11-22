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

def init():
  lucene.initVM()
  
  path = Paths.get("index/")
  indexDir = SimpleFSDirectory(path)

  writerConfig = IndexWriterConfig(StandardAnalyzer())
  return IndexWriter(indexDir, writerConfig)
 
if __name__ == "__main__":
  writer = init()
  
  for folder in os.listdir('data/dates/'):
    path = f'data/dates/{folder}/clusters/'
    if os.path.exists(path):
      for sub_folder in os.listdir(path):
        with open(f'{path}{sub_folder}', 'r') as cluster_file:
          text = cluster_file.read()
          print(text)
          
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
