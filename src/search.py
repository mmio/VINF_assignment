import sys
import lucene
 
from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader, DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
 
if __name__ == "__main__":
    lucene.initVM()
    analyzer = StandardAnalyzer()
    reader = DirectoryReader.open(SimpleFSDirectory(Paths.get("index/")))
    searcher = IndexSearcher(reader)
 
    query = QueryParser("content", analyzer).parse("sears")
    MAX = 1000
    hits = searcher.search(query, MAX)
 
    for hit in hits.scoreDocs:
        print(hit.score, hit.doc, hit.toString())
        doc = searcher.doc(hit.doc)
        print(doc.get("cluster_id"))
