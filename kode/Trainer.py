import multiprocess
from multiprocess import Pool
from nltk.tokenize import word_tokenize
import sys
from time import time
from functools import partial 
from skipgram import SkipGram
from DARV import DARV
import numpy as np

class MultiCoreTrainer:
  def __init__(self):
    self.skipgram_model = SkipGram()
    
  def __create_model(self, n, r, s, h):
    return DARV(n, r, s, h)
    
  def _train_a_model(self, context, n, r, s, h, data):
    skipgram_data = self.skipgram_model.corpus_transform(data, context_count=context)
    model = self.__create_model(n, r, s, h)
    model.partial_fit(skipgram_data, restore=False)
    return model



class DARVTrainer:
  def __init__(self, model):
    self.n = model.n
    self.r = model.r
    self.s = model.s
    self.h = model.h
    self.skipgram_model = SkipGram()
    self.check_point = 0

  def __slice(self, data, worker):
    def chunks(lst, n):
      for i in range(0, len(lst), n):
        yield lst[i:i + n]
    
    return list(chunks(data, int(len(data)/worker)))

  def _train_a_model(self, context, n, r, s, h, data):
      multi_core_trainer = MultiCoreTrainer()
      return multi_core_trainer._train_a_model(context, n, r, s, h, data)

  def __train(self, worker, data, context, num_data_per_worker):
    if worker <= 1 or len(data)<num_data_per_worker:
      result = []
      result.append(self._train_a_model(context, self.n, self.r, self.s, self.h, data))
    else:
      with Pool(worker) as pool:
        sliced_data = self.__slice(data, worker)
        del data
        func = partial(self._train_a_model, context, self.n, self.r, self.s, self.h)
        result = pool.map(func, sliced_data)
    return result
    

  def train(self, filename, model, context=3,  worker=multiprocess.cpu_count(), num_data_per_worker=5000, max_data=10000):
    print('index', 'elapsed_time', 'cum_time')
    with open(filename, 'r', encoding='cp1252') as file:
      data = []
      start = time()
      iter_start = start
      for i, line in enumerate(file):
        data.append(line)
        if i < self.check_point:
          continue
        if len(data) >= worker * num_data_per_worker:
          result = self.__train(worker, data, context, num_data_per_worker)
          for r in result:
            model.merge(r, restore=False) 
          finish = time()
          print(i, finish - iter_start, finish-start)
          iter_start = time()
          data = []
          self.check_point = i
        if i >= self.check_point + max_data and max_data > 1:
          break
      print(i, len(data))
      result = self.__train(worker, data, context, num_data_per_worker)
      for r in result:
        model.merge(r, restore=False)
      model.finish_fit()
      return model