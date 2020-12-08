import spacy
import langdetect
from spacy.tokens import Doc

Doc.set_extension("language", default="unknown")

def language_detector(doc):
    try:
        result = langdetect.detect_langs(doc.text)
        result = str(result[0])[:2]
    except:
        result = 'unknown'

    doc._.language = result

    return doc

def get_pipe():
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(language_detector, name="language", last=True)

    return nlp

def get_tokenizer(nlp):
    return nlp.Defaults.create_tokenizer(nlp)
