from pdb import set_trace as T
from itertools import chain
import numpy as np

from forge.blade.lib.log import Blob

def discountRewards(rewards, gamma=0.99):
   # Untested function
   rets, N = [], len(rewards)
   discounts = np.array([gamma**i for i in range(N)])
   rewards = np.array(rewards)
   for idx in range(N):
      rets.append(sum(rewards[idx:]*discounts[:N-idx]))
   return rets


def sumReturn(rewards):
   """
   Returns a flattened array of rewards.
   """
   # TODO: Unused function.
   return [sum(rewards) for e in rewards]


def mergeRollouts(rollouts):
   atnArgs = [rollout.atnArgs for rollout in rollouts]
   vals    = [rollout.values for rollout in rollouts]
   rets    = [rollout.returns for rollout in rollouts]

   atnArgs = list(chain(*atnArgs))
   atnArgs = list(zip(*atnArgs))
   vals    = list(chain(*vals))
   rets    = list(chain(*rets))

   return atnArgs, vals, rets


class Rollout:
   """
   Aggregates results of multiple steps for a single entity in a Blob,
   recorded by a Feather.
   """
   def __init__(self, returnf=discountRewards):
      self.atnArgs = []
      self.values = []
      self.rewards = []
      self.pop_rewards = [] # TODO: unused
      self.returnf = returnf
      self.feather = Feather()

   def step(self, atnArgs, val, reward, stim, ent):
      """
      Collects stats and updates its underlying Feather.
      """
      self.atnArgs.append(atnArgs)
      self.values.append(val)
      self.rewards.append(reward)

      # TODO: Rollout duplicates many statistics recorded in Feather.
      self.feather.scrawl(atnArgs, val, reward, stim, ent)

   def finish(self):
      # self.rewards[-1] = -1 # NOTE: Something to do with numerical stability.
      self.returns = self.returnf(self.rewards)
      self.lifetime = len(self.rewards)
      self.feather.finish()


class Feather:
   """
   Logger class for a single Rollout. Feather records values for Blob to be
   pickled, whereas Rollout statistics are used in the optimizer.
   """
   def __init__(self):
      self.expMap = set()
      self.blob = Blob()

   def scrawl(self, atnArgs, val, reward, stim, ent):
      """
      Records stim, ent, val, and reward data for a step.
      """
      self.blob.annID = ent.annID
      tile = self.tile(stim)
      self.move(tile, ent.pos)
      #self.action(arguments, atnArgs)
      self.stats(val, reward)

   def tile(self, stim):
      R, C = stim.shape
      rCent, cCent = R//2, C//2
      tile = stim[rCent, cCent]
      return tile

   def action(self, arguments, atnArgs):
      # TODO: Doesn't do anything yet
      move, attk = arguments
      moveArgs, attkArgs, _ = atnArgs
      moveLogits, moveIdx = moveArgs
      attkLogits, attkIdx = attkArgs

   def move(self, tile, pos):
      """
      Updates tile counts with the position of the entity.
      """
      tile = type(tile.state)
      if pos not in self.expMap:
         self.expMap.add(pos)
         self.blob.unique[tile] += 1
      self.blob.counts[tile] += 1

   def stats(self, value, reward):
      self.blob.rewards.append(reward)
      self.blob.values.append(float(value))

   def finish(self):
      self.blob.finish()
