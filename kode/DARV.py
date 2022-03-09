import numpy as np
from scipy.spatial import distance
from time import time
from hashlib import md5, sha256

def hash_lib_to_h(hash_func):
    return lambda x : int(hash_func(x.encode('utf-8')).hexdigest(), 16)

h_md5 = hash_lib_to_h(md5)
h_sha256 = hash_lib_to_h(sha256)


class DARV:
  """
    Distributed Averaged Random Vector
  """
  def __init__(self, n=300, r=np.random.randn, s=np.random.seed, h=[h_md5, h_sha256]):
    self.n = n
    self.r = r
    self.s = s
    self.h = h
    self.word_vector = {}
    self.word_count = {}
    self.restored = False

  def restore(self):
    #if self.restored:
     # raise Exception("Trying to restored restored model");

    #restore word vector
    for key in self.word_vector.keys():
      self.word_vector[key] *= self.word_count[key]
    self.restored = True

  def average(self):
    #if not self.restored:
    #  raise Exception("Trying to average already averaged model");
    
    #average
    for key in self.word_vector.keys():
      self.word_vector[key] /= self.word_count[key]
    self.restored = False
    

  def _get_context_vector(self, context):
    vec = np.zeros(self.n)
    for h in self.h:
      self.s(h(context) % 4294967296) #seed
      vec += self.r(self.n) #random vector  
    vec /= len(self.h)
    return vec

  def word_context_count(self, data):
    count = {}
    for word, context in data:
      count.setdefault(word, {})
      count[word].setdefault(context, 0)
      count[word][context] += 1
    return count

  def partial_fit(self, word_context_pairs, restore=True):
    #restore
    if restore:
      self.restore()
     
    cache = Cache()
    '''
    #word-context count
    word_context_count = self.word_context_count(word_context_pairs)
    del word_context_pairs
    #train
    for word in word_context_count.keys():
      for context in word_context_count[word].keys():
        count = word_context_count[word][context]
        self.word_vector.setdefault(word, np.zeros(self.n)) #zeros
        self.word_count.setdefault(word, 0) #zero count
        self.s(self.h(context) % 4294967296) #seed
        context_vector = self.r(self.n) #random vector
        self.word_count[word] += count
        self.word_vector[word] += count * context_vector
    '''
    for word, context in word_context_pairs:  
      self.word_vector.setdefault(word, np.zeros(self.n)) #zeros
      self.word_count.setdefault(word, 0) #zero count
      context_vector = cache.get(context)
      if context_vector is None:
          context_vector = self._get_context_vector(context)
          cache.add(context, context_vector)
      self.word_count[word] += 1
      self.word_vector[word] += context_vector
    
    del cache
    #average
    if restore:
      self.average()


  def merge(self, model, restore=True):
    if restore:
      self.restore()
      model.restore()

    for key in model.word_vector.keys():
      self.word_vector.setdefault(key, np.zeros(self.n))
      self.word_count.setdefault(key, 0)
      self.word_vector[key] += model.word_vector[key]
      self.word_count[key] += model.word_count[key]

    if restore:
      self.average()
      model.average()

  def finish_fit(self):
    self.average()

  def dissimilarity(self, w1, w2, metric='euclidean'):
    if w1 not in self.word_vector.keys() or w2 not in self.word_vector.keys():
      return None

    w1 = self.word_vector[w1]
    w2 = self.word_vector[w2]
    
    return distance.cdist([w1], [w2], metric)

  def most_similar(self, word, n=10, metric='euclidean'):
    if word not in self.word_vector.keys():
      return None

    vector = self.word_vector[word]
    l = []
    for key in self.word_vector.keys():
      l.append((key, distance.cdist([vector], [self.word_vector[key]], metric=metric)))

    l = sorted(l, key=lambda x : x[1])
    return l[:n+1];


class Cache:
    def __init__(self, max_cache=10000):
        self.cache = {}
        self.index = []
        self.max = max_cache
        
    def add(self, word, vector):
        while len(self.cache) > self.max:
            key = self.index.pop(0)
            del self.cache[key]
        self.index.append(word)
        self.cache[word] = vector
        
    def get(self, word):
        return self.cache.get(word, None)
    
    def reset(self):
        self.cache = {}
        self.index = []