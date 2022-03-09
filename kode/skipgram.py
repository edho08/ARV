import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re

class AbstractContext:
  def __init__(self):
    pass	

  def _abstract_op(self, context, word, dist):
    pass	
  
  def transform(self, sentence, distance=False, context_count=1):
    tokens = word_tokenize(sentence)
    for i_w in range(len(tokens)):
      for i_b in range(1, context_count+1):
        i_c = i_w - i_b
        if i_c > 0:
          if not distance:
            yield self._abstract_op(tokens[i_c], tokens[i_w])
          else:
            yield self._abstract_op(tokens[i_c], tokens[i_w], i_b)
      for i_a in range(1, context_count+1):
        i_c = i_w + i_a
        if i_c < len(tokens):
          if not distance:
            yield self._abstract_op(tokens[i_c], tokens[i_w])
          else:
            yield self._abstract_op(tokens[i_c], tokens[i_w], i_a)

  def doc_transform(self, doc, context_count=1, unique=False, distance=False):
    map_sos_eos = lambda x : re.sub("[.?!]", "[EOS]", x)
    doc = map_sos_eos(doc)
    sentences = doc.split('[EOS]')
    t = []
    for sentence in sentences:
      t.extend(list(self.transform(sentence, context_count=context_count, distance=distance)))
    if unique:
      return list(set(t))
    else:
      return t
  """
  def corpus_transform(self, corpus, context_count=1, unique=False, distance=False):
    t = []
    for doc in corpus:
      t.extend(self.doc_transform(doc, context_count, unique, distance=distance))
    if unique:
      return list(set(t))
    else:
      return t
  """
  def corpus_transform(self, corpus, context_count=1, unique=False, distance=False):
    t = []
    for doc in corpus:
      map_sos_eos = lambda x : re.sub("[.?!]", "[EOS]", x)
      doc = map_sos_eos(doc)
      sentences = doc.split('[EOS]')
      for sentence in sentences:
          tokens = word_tokenize(sentence)
          for i_w in range(len(tokens)):
            for i_b in range(1, context_count+1):
              i_c = i_w - i_b
              if i_c > 0:
                if not distance:
                  yield self._abstract_op(tokens[i_c], tokens[i_w])
                else:
                  yield self._abstract_op(tokens[i_c], tokens[i_w], i_b)
            for i_a in range(1, context_count+1):
              i_c = i_w + i_a
              if i_c < len(tokens):
                if not distance:
                  yield self._abstract_op(tokens[i_c], tokens[i_w])
                else:
                  yield self._abstract_op(tokens[i_c], tokens[i_w], i_a)
                    
  def to_vector(self, t1, t2):
    return (self.vocab.get_one_hot_dic(t1), self.vocab.get_one_hot_dic(t2))        
    

class SkipGram(AbstractContext):
  def _abstract_op(self, context, word, dist=0):
    if dist == 0:
      return (word, context)
    else:
      return (word, context, dist)