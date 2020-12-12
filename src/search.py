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

    subfolder_index_pairs = [
        ('cluster_w2v', 'index_w2v'),
        ('cluster_w2v_n', 'index_w2v_n'),
        ('cluster_tfidf', 'index_tfidf'),
        ('cluster_tfidf_n', 'index_tfidf_n')
    ]

    search_term = 'dominik'

    for cluster, index in subfolder_index_pairs:
        print(f'searching in {index}')
        path = Paths.get(index)
        
        reader = DirectoryReader.open(SimpleFSDirectory(path))
        searcher = IndexSearcher(reader)
 
        query = QueryParser("content", analyzer).parse(search_term)
        MAX = 1000000
        hits = searcher.search(query, MAX)
 
        for hit in hits.scoreDocs:
            # print(hit.score, hit.doc, hit.toString())
            doc = searcher.doc(hit.doc)

            day = doc.get('day')
            cluster = doc.get('cluster')
            print(day, cluster)

