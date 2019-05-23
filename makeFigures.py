import argparse
import sys, json
from itertools import groupby
from collections import defaultdict
import pickle
import os
from pdb import set_trace as T

from matplotlib import pyplot as plt
import numpy as np

from figures import logs as loglib
from forge.blade.lib.enums import Neon, Color256
from forge.blade.lib.log import InkWell
import experiments


#########################################
# Utils for tinkering with data for plots
#########################################

def meanfilter(X, n=0):
   """
   Unused. Uses a mean filter of size n over X.
   """
   ret = []
   for idx in range(len(X) - n):
      val = np.mean(X[idx:(idx+n)])
      ret.append(val)
   return ret

def compress(x, split):
   """
   Compresses data along an axis.
   """
   rets, idxs = [], []
   if split == 'train':
      n = 1 + len(x) // 20
   else:
      # TODO: what to do differently here?
      n = 1 + len(x) // 20

   for idx in range(0, len(x) - n, n):
      rets.append(np.mean(x[idx:(idx+n)]))
      idxs.append(idx)
   return 10 * np.array(idxs), rets

def flipDict(popLogs):
   """
   Helper to rearrange the dimensions of a dictionary.
   """
   ret = defaultdict(dict)
   for annID, logs in popLogs.items():
      for key, log in logs.items():
         if annID not in ret[key]:
            ret[key][annID] = []
         if type(log) != list:
            ret[key][annID].append(log)
         else:
            ret[key][annID] += log
   return ret

def mergePops(blobs, groupFunc):
   """
   Pool blob info according to the groupFunc.
   """
   mergedBlobs = defaultdict(list)
   for blob in blobs:
      mergedBlobs[groupFunc(blob.annID)].append(blob)
   pops = defaultdict(list)
   for groupID, blobList in mergedBlobs.items():
      pops[groupID] += list(blobList)
   return pops

def makeGroupFunc(config, form='single'):
   """
   Returns a function mapping a population to a group.
   """
   assert form in 'pops single split'.split()
   if form == 'pops':
      # Plot the statistics for each individual population
      return lambda x: x

   elif form == 'single':
      # Aggregate all populations into a single group
      return lambda x: 0

   elif form == 'split':
      # Split populations into groups - TODO
      pop1 = dict((idx, 0) for idx in range(config.NPOP1))
      pop2 = dict((idx, 0) for idx in range(config.NPOP2))
      return {**pop1, **pop2}


###############################################################
# Methods for plotting statistics over time against populations
###############################################################

def plot(x, idxs, label, idx, path):
   """
   Unused. Plots a single array of data.
   """
   colors = Neon.color12()
   loglib.dark()
   c = colors[idx % 12]
   loglib.plot(x, inds=idxs, label=str(idx), c=c.norm)
   loglib.godsword()
   loglib.save(path + label + '.png')
   plt.close()

def plots(x, label, path, split):
   """
   Generates plots for multiple blobs in a single experiment.
   """
   colors = Neon.color12()
   loglib.dark()

   for i, item in enumerate(x.items()):
      annID, val = item
      c = colors[i % 12]
      idxs, val = compress(val, split)
      loglib.plot(val, inds=idxs, label=str(annID), c=c.norm)

   loglib.godsword()
   # Overwrite title with label
   loglib.labels('Steps', 'Value', 'Projekt: Godsword - %s' % label)
   loglib.save(os.path.join(path, label + '.png'))
   plt.close()

def plotBlobs(blobs, saveDir, groupFunc, split, quiet):
   """
   Sets up plots for a single experiment.
   """
   saveSplitDir = os.path.join(saveDir, split)
   if not os.path.exists(saveSplitDir):
      os.makedirs(saveSplitDir)

   blobs = mergePops(blobs, groupFunc)
   popLogs = {}
   for annID, blobList in blobs.items():
      logs, blobList = {}, list(blobList)
      logs = {**logs, **InkWell.counts(blobList)}
      logs = {**logs, **InkWell.unique(blobList)}
      logs = {**logs, **InkWell.explore(blobList)}
      logs = {**logs, **InkWell.lifetime(blobList)}
      logs = {**logs, **InkWell.reward(blobList)}
      logs = {**logs, **InkWell.value(blobList)}
      popLogs[annID] = logs

   popLogs = flipDict(popLogs)
   for label, val in popLogs.items():
      if not quiet:
         print("\tPlotting", label)
      #val = meanfilter(val, 1+len(val)//100)
      plots(val, str(label), saveSplitDir, split)

def plotExperimentLog(logfname, config, group, quiet):
   """
   Generates plots of all statistics for a single experiment.
   """
   with open(logfname, 'rb') as f:
      blobs = []
      numBlobs = 0
      while True:
         numBlobs += 1
         try:
            blobs += pickle.load(f)
         except EOFError as e:
            break
      print('Blob length: ', numBlobs)

      split = 'test' if config.TEST else 'train'
      groupFunc = makeGroupFunc(config, "single" if group else "pops")
      parent = os.path.join(logfname, os.pardir)
      saveDir = os.path.abspath(os.path.join(parent, os.pardir))
      plotBlobs(blobs, saveDir, groupFunc, split, quiet=quiet)
      print('Log success: ', logfname)


###############
# Main function
###############

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Create figures for an experiment.")
   parser.add_argument("--group", action='store_true')
   parser.add_argument("--quiet", action='store_true')
   args = parser.parse_args()

   logDir  = 'resource/exps'
   logName = 'model/logs.p'
   group = args.group
   quiet = args.quiet

   for (expName, config) in experiments.exps.items():
      try:
         plotExperimentLog(
               os.path.join(logDir, expName, logName), config, group, quiet)
      except Exception as err:
         print(str(err))
