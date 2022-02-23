import stanza
from stanza.server import CoreNLPClient


class Pipeline:
    def __init__(self, port=0, install_corenlp=False):
        if install_corenlp:
            stanza.install_corenlp()
            stanza.download_corenlp_models(model='english', version='4.1.0')
        self.core_nlp = CoreNLPClient(annotators=['tokenize', 'ssplit', 'depparse'], timeout=3000, be_quiet=True,
                                      properties={"ssplit.newlineIsSentenceBreak": "always"},
                                      endpoint="http://localhost:{}".format(9000+port))

    def __call__(self, document):
        try:
            doc = self.core_nlp.annotate(document)
        except:
            return False
        sents = []
        sent_heads = []
        for sent in doc.sentence:
            words = []
            for token in sent.token:
                words.append(token.word.lower())

            heads = [i for i in range(len(words))]
            for edge in sent.basicDependencies.edge:
                heads[edge.target - 1] = edge.source - 1

            sents.append(words)
            sent_heads.append(heads)
        return sents, sent_heads
