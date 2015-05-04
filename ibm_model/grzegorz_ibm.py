# Grzegorz Chrupala
# Code loosely based on https://github.com/kylebgorman/model1/blob/master/m1.py

# Implementation of the online version of EM for IBM model 1.
from __future__ import division
import numpy as np


def bitext(source, target, weighted):
    """
    Return sentence pairs
    """
    print "Processing", len(target), "lines"

    if weighted:
        print "Weighted mode"
        weighted_source = [ [ (i,f) for i,f in enumerate(src) if f > 0.0 ] for src in source  ]
    else:
        weighted_source = [ [ (f,1.0) for f in src ] for src in source ]
    return zip(weighted_source, target)


class M1(object):
    """
    A class wrapping an IBM Model 1 t-table. 
    """
    def __init__(self, source, target, online=True, weighted=False, skip_id=False):
        '''Args
        - source (list) - input text
        - target (list) - output text
        - online (bool) - use online algorithm
        - weighted (bool) - source words are weights
        - skip_id (bool) - ignore self-translations (for monolingual setup)
        '''
        self.source = source
        self.target = target
        self.eps    = 1.0
        self.online = online
        self.weighted = weighted
        self.skip_id = skip_id
        self.data   = bitext(self.source, self.target, weighted)

        # Word co-occurrence counts 
        # self.acounts[(f,e)] counts how many times f was aligned to e
        self.acounts = {}     
        # Word occurrence counts
        # self.tcounts[e] counts how many times any word was aligned to e
        self.tcounts = {}     
        self.f_vocab = set()
        # Static ttable for batch algo
        # ttable[(f,e)] returns the probability that f is a translation of e
        self._ttable = {}
        if not self.online:
            self._initialize_batch()

    def _initialize_batch(self):
        counts = {}
        counter = 0
        print "Initializing batch"

        # compute raw co-occurrence counts
        for (src, tgt) in self.data:
            counter += 1
            for f,w in src:
                print 'At line', counter,'\r',
                for e in tgt:
                    counts[(f,e)] = w
        sums = {}
        for (f, e) in counts.keys():
            sums[e] = sums.get(e, 0.0) + 1.0
        for (f, e) in counts.keys():
            self._ttable[(f,e)] = counts[(f,e)]/sums[e]
    
    def ttable(self, f, e):
        """
        Translation table (smoothed)
        ttable(f, e) returns the probability that f is a translation of e 
        """
        # Translation probabilities are recomputed from expected word counts
        # stored in self.acounts and self.tcounts
        # 
        # t(f|e) = c(f,e)/SUM_f' c(f',e)
        self.f_vocab.add(f)
        return  self.acounts.get((f, e), self.eps) / self.tcounts.get(e, self.eps * len(self.f_vocab))

    def iterate(self):
        if self.online:
            self._iterate_online()
        else:
            self._iterate_batch()

    def _iterate_online(self):
        """
        Iterate once through the data. 
        """
	counter = 0
        for (src, tgt) in self.data:
	   # print src, tgt
	    counter += 1
	    print 'At line', counter, '\r',
            for i,fw in enumerate(src):
                f, w = fw
                # sum of t(f|e') over all e's in current target sentence
                total_f = sum((self.ttable(f, e) for e in tgt))
                for j,e in enumerate(tgt):
                    if i != j or not self.skip_id:
                        # count for (f,e) in current sentence pair
                        c = w * self.ttable(f, e) / total_f 
                        # increment (co) occurrence count tables with current c(f,e)
                        self.acounts[(f, e)] = self.acounts.get((f, e), 0.0) + c
                        self.tcounts[e]      = self.tcounts.get(e, 0.0) + c

            ## M-step is implicit and on demand via self.ttable

    def _iterate_batch(self):
        """
        Perform 1 iteration of EM training
        """
        print "Performing Batch iteration"

        acounts = {}
        tcounts = {}
        total={}
        counter = 0
        ## E-step
        for (src, tgt) in self.data:
            counter += 1
            print 'At sentence', counter, '\r',
            for i,fw in enumerate(src):
                f,w = fw
                total[f] = 0.0
                for j,e in enumerate(tgt):
                    if i != j or not self.skip_id:
                    # compute expectation and preserve it
                        total[f] = total.get(f, 0.0) + self._ttable[(f,e)]
                for j,e in enumerate(tgt):
                    if i != j or not self.skip_id:
                        c = w * self._ttable[(f,e)] / total[f]
                        acounts[(f, e)] = acounts.get((f,e), 0.0) + c
                        tcounts[e] = tcounts.get(e, 0.0) + c 
        ## M-step
        for (f, e) in acounts.keys():
            self._ttable[(f,e)] = acounts[(f, e)] / tcounts[e]
        # Store acounts
        self.acounts = acounts
        

    def translation_table(self):
        """
        Compute and return translation table from internal state
        """
        table = {}
        for ((k1, k2), v) in self.acounts.iteritems():
            inner = table.get(k2, {}) 
            if self.online:
                inner[k1] = self.ttable(k1, k2)
            else:
                inner[k1] = self._ttable[(k1,k2)]
            table[k2] = inner
        return table
