from pdb import set_trace as T
from collections import defaultdict
import numpy as np

from forge import trinity
from forge.ethyr.torch.param import setParameters, zeroGrads
from forge.ethyr.torch import optim
from forge.ethyr.rollouts import Rollout


class Sword:
   def __init__(self, config, args):
      self.config, self.args = config, args
      self.nANN, self.h = config.NPOP, config.HIDDEN
      self.anns  = [trinity.ANN(config) for i in range(self.nANN)]

      self.init = True

      self.networksUsed = set()
      self.updates = defaultdict(Rollout)
      self.rollouts = {} # starts empty, gets populated
      self.ents, self.rewards, self.grads = {}, [], None

      # Ways to initiate backward:
      # keeps track of total lifetimes, before initiating backward()
      self.nGrads = 0
      # a threshold for the max number of Rollouts before initiating backward()
      self.nRollouts = 32

   def backward(self):
      """
      Backward pass through optim.backward().
      """
      ents = self.rollouts.keys()
      anns = [self.anns[idx] for idx in self.networksUsed]

      reward, val, grads, pg, valLoss, entropy = optim.backward(
            self.rollouts,
            anns,
            valWeight=0.25,
            entWeight=self.config.ENTROPY)
      self.grads = dict((idx, grad) for idx, grad in
            zip(self.networksUsed, grads))

      self.blobs = [r.feather.blob for r in self.rollouts.values()]
      self.rollouts = {}
      self.nGrads = 0
      self.networksUsed = set()

   def sendGradUpdate(self):
      """
      Helper for sendUpdate
      """
      grads = self.grads
      self.grads = None
      return grads

   def sendLogUpdate(self):
      """
      Helper for sendUpdate
      """
      blobs = self.blobs
      self.blobs = []
      return blobs

   def sendUpdate(self):
      """
      Used in forge/blade/core/realm.py to propagate updates.
      """
      if self.grads is None:
          return None, None
      return self.sendGradUpdate(), self.sendLogUpdate()

   def recvUpdate(self, update):
      for idx, paramVec in enumerate(update):
         setParameters(self.anns[idx], paramVec)
         zeroGrads(self.anns[idx])

   def collectStep(self, entID, atnArgs, val, reward, stim, ent):
      if self.config.TEST:
          return
      self.updates[entID].step(atnArgs, val, reward, stim, ent)

   def collectRollout(self, entID, ent, backwardThres=100*32):
      """
      Finishes a Rollout, then moves it from self.updates to self.rollouts.
      Initiates backward() if the total lifetimes of the rollouts is above
      a threshold.
      """
      assert entID not in self.rollouts
      rollout = self.updates[entID]
      rollout.finish()
      self.nGrads += rollout.lifetime
      self.rollouts[entID] = rollout
      del self.updates[entID]

      # assert ent.annID == (hash(entID) % self.nANN)
      self.networksUsed.add(ent.annID)

      # Option 1: fixed number of rollouts
      # if len(self.rollouts) >= self.nRollouts:
      # Option 2: fixed number of gradients
      if self.nGrads >= backwardThres:
         self.backward()

   def decide(self, ent, stim, popCounts, coop=True):
      reward, entID, annID = 1.0, ent.entID, ent.annID
      if coop:
         # Cooperation experiment: increase reward by proportion of living
         # agents in its own population
         coopScale = 0.1
         totalCounts = float(popCounts.sum())
         reward += coopScale * popCounts[annID] / totalCounts

      # Decide on an action
      action, arguments, atnArgs, val = self.anns[annID](ent, stim)

      # Updates the rollout
      self.collectStep(entID, atnArgs, val, reward, stim, ent)

      return action, arguments, float(val)
