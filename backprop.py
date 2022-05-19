# TODO : Improve the computation speed by using the Numpy module,
# instead of using default python matrices

import math
import random
import string

class NN:
  def __init__(self, NI, NH, NO):
    self.newSigmoid = False
    # number of nodes in layers
    self.ni = NI #+1 #for bias
    self.nh = NH
    self.no = NO

    # initialize node-activations
    self.ai, self.ah, self.ao = [], [], []
    self.ai = [1.0]*self.ni
    self.ai[self.ni-1] =  random.uniform(-1, 1)
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no

    # create node weight matrices
    self.wi = makeMatrix (self.ni, self.nh)
    self.wo = makeMatrix (self.nh, self.no)
    # create bias
    self.b = [0.0]*(self.nh + self.no)
    # momentum for the bias
    self.bmom = [0.0] * (self.nh + self.no)
    # initialize node weights to random vals
    randomizeMatrix ( self.wi, -1., 1. )
    randomizeMatrix ( self.wo, -1., 1. )
    # create last change in weights matrices for momentum
    self.ci = makeMatrix (self.ni, self.nh)
    self.co = makeMatrix (self.nh, self.no)

  def runNN (self, inputs):
    #if len(inputs) != self.ni-1:
    # Update after changing how the bias are handled
    if len(inputs) != self.ni:
      print("incorrect number of inputs")

#    for i in range(self.ni-1):
    for i in range(self.ni):
      self.ai[i] = inputs[i]

    for j in range(self.nh):
      #sum = 0.0
      # init with the bias
      sum = self.b[j]
      for i in range(self.ni):
        sum +=( self.ai[i] * self.wi[i][j] )
      self.ah[j] = self.sigmoid (sum)

    for k in range(self.no):
      #sum = 0.0
      # init with the bias
      sum = self.b[-k-1]
      for j in range(self.nh):
        sum +=( self.ah[j] * self.wo[j][k] )
      self.ao[k] = self.sigmoid (sum)

    return self.ao



  #def backPropagate (self, targets, N, M):
  def backPropagate (self, grad, N, M):
    # http://www.youtube.com/watch?v=aVId8KMsdUU&feature=BFa&list=LLldMCkmXl4j9_v0HeKdNcRA

    # calc output deltas
    # we want to find the instantaneous rate of change of ( error with respect to weight from node j to node k)
    # output_delta is defined as an attribute of each ouput node. It is not the final rate we need.
    # To get the final rate we must multiply the delta by the activation of the hidden layer node in question.
    # This multiplication is done according to the chain rule as we are taking the derivative of the activation function
    # of the ouput node.
    # dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
    output_deltas = [0.0] * self.no
    for k in range(self.no):
      #error = targets[k] - self.ao[k]
      output_deltas[k] =  grad[k] * self.dsigmoid(self.ao[k])

    # update output weights
    for j in range(self.nh):
      for k in range(self.no):
        # output_deltas[k] * self.ah[j] is the full derivative of dError/dweight[j][k]
        change = output_deltas[k] * self.ah[j]

        self.wo[j][k] += N*change + M*self.co[j][k]
        self.co[j][k] = change

    # update bias on output layer
    for k in range(self.no):
        # no momentum
        change = output_deltas[k]
        self.b[-k-1] +=  N*change + M*self.bmom[-k-1]
        self.bmom[-k-1] = change

    # calc hidden deltas
    hidden_deltas = [0.0] * self.nh
    for j in range(self.nh):
      error = 0.0
      for k in range(self.no):
        error += output_deltas[k] * self.wo[j][k]
      hidden_deltas[j] = error * self.dsigmoid(self.ah[j])

      # update bias on hidden layer
      change = hidden_deltas[j]
      self.b[j] += N*change + M*self.bmom[j]
      self.bmom[j] = change

    #update input weights
    for i in range (self.ni):
      for j in range (self.nh):
        change = hidden_deltas[j] * self.ai[i]
        #print 'activation',self.ai[i],'synapse',i,j,'change',change
        self.wi[i][j] += N*change + M*self.ci[i][j]
        self.ci[i][j] = change

    # calc combined error
    # 1/2 for differential convenience & **2 for modulus
#    error = 0.0
#    for k in range(len(targets)):
#      error = 0.5 * (targets[k]-self.ao[k])**2
#    return error
    return False

  def random_update(self, random_ratio):
      for j in range(self.nh):
          for k in range(self.no):
              maximum_perturbation = random_ratio * self.wo[j][k]
              self.wo[j][k] +=random.uniform(-maximum_perturbation, maximum_perturbation)

      for i in range(self.ni):
          for j in range(self.nh):
              maximum_perturbation = random_ratio * self.wi[i][j]
              self.wi[i][j] +=random.uniform(-maximum_perturbation, maximum_perturbation)


  def weights(self):
    print('Input weights:')
    for i in range(self.ni):
      print(self.wi[i])
    print
    print('Output weights:')
# 12/12 changement de nh pour no sur ligne d'en dessous
    for j in range(self.no):
      print(self.wo[j])
    print('')

  def sigmoid (self, x):
      if self.newSigmoid:
          return 1.7159 * math.tanh((2/3) * x)
      else:
          return math.tanh(x)

    # the derivative of the sigmoid function in terms of output
    # proof here:
    # http://www.math10.com/en/algebra/hyperbolic-functions/hyperbolic-functions.html
  def dsigmoid (self, y):
      if self.newSigmoid:
          return (2/3) * (1.7159 - y**2)
      else:
          return 1 - y**2

def makeMatrix ( I, J, fill=0.0):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m

def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = random.uniform(a,b)