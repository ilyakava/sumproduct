import numpy as np
import pdb
from math import isinf

class Node:
  def __init__(self, name):
    self.connections = []
    self.inbox = {} # messages recieved
    self.name = name

  def append(self, to_node):
    """
    Mutates the to AND from node!
    """
    self.connections.append(to_node)
    to_node.connections.append(self)

  def deliver(self, step_num, mu):
    """
    Ensures that inbox is keyed by a step number
    """
    if self.inbox.get(step_num):
      self.inbox[step_num].append(mu)
    else:
      self.inbox[step_num] = [mu]

class Factor(Node):
  """
  NOTE: For the Factor nodes in the graph, it will be assumed
  that the connections are created in the same exact order
  as the potentials' dimensions are given
  """
  def __init__(self, name, potentials):
    self.p = potentials
    Node.__init__(self, name)

  def make_message(self, recipient):
    """
    Does NOT mutate the Factor node!

    NOTE that using the log rule before 5.1.42 in BRML by David
    Barber, that the product is actually computed via a sum of logs.

    Steps:
    1. reformat mus to all be the same dimension as the factor's
    potential and take logs, mus -> lambdas
    2. find a max_lambda (element wise maximum)
    3. sum lambdas, and subtract the max_lambda once
    4. exponentiate the previous result, multiply by exp of max_lambda
    and run summation to sum over all the states not in the recipient
    node

    The treatment of the max_lambda differs here from 5.1.42, which
    incorrectly derived from 5.1.40 (you cannot take lambda* out of
    the log b/c it is involved in a non-linear operation)
    """
    if not len(self.connections) == 1:
      unfiltered_mus = self.inbox[max(self.inbox.keys())]
      mus = [mu for mu in unfiltered_mus if not mu.from_node == recipient]
      all_mus = [self.reformat_mu(mu) for mu in mus]
      lambdas = [np.log(mu) for mu in all_mus]
      max_lambda_nan = reduce(lambda a,e: np.maximum(a,e), lambdas)
      max_lambda = np.nan_to_num(max_lambda_nan)
      result = reduce(lambda a,e: a + e, lambdas) - max_lambda
      product_output2 = np.multiply(self.p, np.exp(result))
      product_output = np.multiply(product_output2, np.exp(max_lambda))
      return self.summation(product_output, recipient)
    else:
      return self.summation(self.p, recipient)

  def reformat_mu(self, mu):
    """
    Returns the given mu's val reformatted to be the same
    dimensions as self.p, ensuring that mu's values are
    expanded in the correct axes.

    The identity of mu's from_node is used to decide which axis
    the mu's val should be expaned in to fit self.p

    Example:

    # self.p (dim order: x3, x4, then x2)
    np.array([
      [
        [0.3,0.5,0.2],
        [0.1,0.1,0.8]
      ],
      [
        [0.9,0.05,0.05],
        [0.2,0.7,0.1]
      ]
    ])

    # mu
    x3 = np.array([0.2, 0.8])
    which_dim = 0 # the dimension which x3 changes in self.p
    dims = [2, 2, 3]

    # desired output
    np.array([
      [
        [0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2]
      ],
      [
        [0.8, 0.8, 0.8],
        [0.8, 0.8, 0.8]
      ]
    ])
    """
    dims = self.p.shape
    states = mu.val
    which_dim = self.connections.index(mu.from_node) # raises err
    assert dims[which_dim] is len(states)

    acc = np.ones(dims)
    for coord in np.ndindex(dims):
      i = coord[which_dim]
      acc[coord] *= states[i]
    return acc

  def summation(self, p, node):
    """
    Does NOT mutate the factor node.

    Sum over all states not in the node.
    Similar to reformat_mu in strategy.
    """
    dims = p.shape
    which_dim = self.connections.index(node)
    out = np.zeros(node.size)
    assert dims[which_dim] is node.size
    for coord in np.ndindex(dims):
      i = coord[which_dim]
      out[i] += p[coord]
    return out

class Variable(Node):
  def __init__(self, name, size):
    self.size = size
    Node.__init__(self, name)

  def marginal(self):
    """
    Life saving normalizations:

    sum_logs - max(sum_logs) <- before exponentiating
    and rem_inf
    """
    if len(self.inbox):
      mus = self.inbox[max(self.inbox.keys())]
      log_vals = [np.log(mu.val) for mu in mus]
      valid_log_vals = [self.rem_inf(lv) for lv in log_vals]
      sum_logs = reduce(lambda a, e: a+e, valid_log_vals)
      valid_sum_logs = sum_logs - max(sum_logs) # IMPORANT!
      prod = np.exp(valid_sum_logs)
      return prod / sum(prod) # normalize
    else:
      # first time called: uniform
      return np.ones(self.size) / self.size

  def latex_marginal(self):
    """
    same as marginal() but returns a nicely formatted latex string
    """
    data = self.marginal()
    data_str = ' & '.join([str(d) for d in data])
    tabular = '|' + ' | '.join(['l' for i in range(self.size)]) + '|'
    return ("$$p(\mathrm{" + self.name + "}) = \\begin{tabular}{" +
      tabular +
      '} \hline' +
      data_str +
      '\\\\ \hline \end{tabular}$$')

  @staticmethod
  def rem_inf(arr):
    """
    If needed, remove infinities (specifically, negative
    infinities are likely to occur)
    """
    if isinf(sum(arr)):
      return np.array([0 if isinf(number) else number for number in arr])
    else:
      return np.array(arr)

  def make_message(self, recipient):
    """
    Follows log rule in 5.1.38 in BRML by David Barber
    b/c of numerical issues
    """
    if not len(self.connections) == 1:
      unfiltered_mus = self.inbox[max(self.inbox.keys())]
      mus = [mu for mu in unfiltered_mus if not mu.from_node == recipient]
      log_vals = [np.log(mu.val) for mu in mus]
      return np.exp(reduce(lambda a,e: a+e, log_vals))
    else:
      return np.ones(self.size)

class Mu:
  """
  An object to represent a message being passed
  a to_node attribute isn't needed since that will be clear from
  whose inbox the Mu is sitting in
  """
  def __init__(self, from_node, val):
    self.from_node = from_node
    # this normalization is necessary
    self.val = val.flatten() / sum(val.flatten())
