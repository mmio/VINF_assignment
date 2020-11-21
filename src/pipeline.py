import spacy
from spacy.tokens import Doc
# from spacy_hunspell import spaCyHunSpell
# from spacy_langdetect import LanguageDetector

import langdetect

Doc.set_extension("ld", default="unknown")

def ld(doc):
    try:
        result = langdetect.detect_langs(doc.text)
        result = str(result[0])[:2]
    except:
        result = 'unknown'

    doc._.ld = result

    return doc

def get_pipe():
    nlp = spacy.load("en_core_web_lg")
    # nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    # nlp.add_pipe(spaCyHunSpell(nlp, ('en_US.dic', 'en_US.aff')))
    nlp.add_pipe(ld, name="ld", last=True)

    # print(nlp.pipe_names)  # ['tagger', 'parser', 'ner', 'print_info']

    return nlp

def get_tokenizer(nlp):
    return nlp.Defaults.create_tokenizer(nlp)
