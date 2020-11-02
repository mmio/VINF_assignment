import spacy

# def my_component(doc):
#     print("After tokenization, this doc has {} tokens.".format(len(doc)))
#     print("The part-of-speech tags are:", [token.pos_ for token in doc])
#     if len(doc) < 10:
#         print("This is a pretty short document.")
#     return doc

def get_pipe():
    nlp = spacy.load("en_core_web_lg")
    # nlp.add_pipe(my_component, name="print_info", last=True)
    # print(nlp.pipe_names)  # ['tagger', 'parser', 'ner', 'print_info']
    return nlp

def get_tokenizer(nlp):
    return nlp.Defaults.create_tokenizer(nlp)
