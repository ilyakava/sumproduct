import numpy as np
from nodes_and_messages import Variable, Factor, Mu
import pdb

class FactorGraph:
  nodes = {}
  silent = False

  def __init__(self, first_node=None, silent=False):
    if silent:
      self.silent = silent
    if first_node:
      self.nodes[first_node.name] = first_node

  def add(self, node):
    assert node not in self.nodes
    self.nodes[node.name] = node

  def connect(self, name1, name2):
    # no need to assert since dict lookup will raise err
    self.nodes[name1].append(self.nodes[name2])

  def append(self, from_node_name, to_node):
    assert from_node_name in self.nodes
    tnn = to_node.name
    # add the to_node to the graph if it is not already there
    if not (self.nodes.get(tnn, 0)):
      self.nodes[tnn] = to_node
    self.nodes[from_node_name].append(self.nodes[tnn])
    return self

  def leaf_nodes(self):
    return [node for node in self.nodes.values() if len(node.connections) ==  1]

  def observe(self, name, state):
    """
    Mutates the factors connected to Variable with name!

    As described in Barber 5.1.3. But instead of multiplying
    factors with an indicator/delta_function to account for
    an observation, the factor node loses the dimensions for
    unobserved states, and then the connection to the observed
    variable node is severed (although it remains in the graph
    to give a uniform marginal when asked).
    """
    node = self.nodes[name]
    assert isinstance(node, Variable)
    assert node.size >= state
    for factor in [c for c in node.connections if isinstance(c, Factor)]:
      delete_axis = factor.connections.index(node)
      delete_dims = range(node.size)
      delete_dims.pop(state - 1)
      sliced = np.delete(factor.p, delete_dims, delete_axis)
      factor.p = np.squeeze(sliced)
      factor.connections.remove(node)
      assert len(factor.p.shape) is len(factor.connections)
    node.connections = [] # so that they don't pass messages

  def export_marginals(self):
    return dict([
      (n.name, n.marginal())
      for n in self.nodes.values()
      if isinstance(n, Variable)])

  @staticmethod
  def compare_marginals(m1, m2):
    """
    For testing the difference between marginals across a graph at
    two different iteration states, in order to declare convergence.
    """
    assert not len(np.setdiff1d(m1.keys(), m2.keys()))
    return sum([sum(np.absolute(m1[k] - m2[k])) for k in m1.keys()])

  def compute_marginals(self, max_iter=500, tolerance=1e-6):
    """
    sum-product algorithm

    Mutates nodes by adding in the messages passed into their
    'inbox' instance variables. It does not change the potentials
    on the Factor nodes.

    Using the "Asynchronous Parallel Schedule" from Sudderth lec04
    slide 11 after an initialization step of Variable nodes sending
    all 1's messages:
    - At each iteration, all nodes compute all outputs from all
    current inputs. Factors-Variables and then Variables-Factors
    - Iterate until convergence.

    This update schedule is best suited for loopy graphs. It ends
    up working best as a max sum-product algorithm as high
    probabilities dominate heavily when the tolerance is very small
    """
    # for keeping track of state
    epsilon = 1
    step = 1
    # for testing convergence
    cur_marginals = self.export_marginals()
    # initialization
    for node in self.nodes.values():
      if isinstance(node, Variable):
        message = Mu(node, np.ones(node.size))
        for recipient in node.connections:
          recipient.deliver(step, message)

    # propagation (w/ termination conditions)
    while (step < max_iter) and tolerance < epsilon:
      last_marginals = cur_marginals
      step += 1
      if not self.silent:
        print 'epsilon: ' + str(epsilon) + ' | ' + str(step) + '-'*20
      factors = [n for n in self.nodes.values() if isinstance(n, Factor)]
      variables = [n for n in self.nodes.values() if isinstance(n, Variable)]
      senders = factors + variables
      for sender in senders:
        next_recipients = sender.connections
        for recipient in next_recipients:
          if not self.silent:
            print sender.name + ' -> ' + recipient.name
          val = sender.make_message(recipient)
          message = Mu(sender, val)
          recipient.deliver(step, message)
      cur_marginals = self.export_marginals()
      epsilon = self.compare_marginals(cur_marginals, last_marginals)
    if not self.silent:
      print 'X'*50
      print 'final epsilon after ' + str(step) + ' iterations = ' + str(epsilon)

  def brute_force(self):
    """
    Main strategy of this code was gleaned from:
    http://cs.brown.edu/courses/cs242/assignments/hw1code.zip

    # first compute the full joint table
    - create a joint accumulator for N variables that is N dimensional
    - iterate through factors
      - for each factor expand probabilities into dimensions of the joint
      table
        - create a factor accumulator that is N dimensional
        - for each coord in the joint table, look at the states of the
        vars that are in the factor's potentials, and add in the log of
        that probability
    - exponentiate and normalize
    # then compute the marginals
    - iterate through variables
      - for each variable sum over all other variables
    """
    variables = [v for v in self.nodes.values() if isinstance(v, Variable)]

    var_dims = [v.size for v in variables]
    N = len(var_dims)
    assert N < 32, "max number of vars for brute force is 32 (numpy's matrix dim limit)"
    log_joint_acc = np.zeros(var_dims)
    for factor in [f for f in self.nodes.values() if isinstance(f, Factor)]:
      # dimensions that will matter for this factor
      which_dims = [variables.index(v) for v in factor.connections]
      factor_acc = np.ones(var_dims)
      for joint_coord in np.ndindex(tuple(var_dims)):
        factor_coord = tuple([joint_coord[i] for i in which_dims])
        factor_acc[joint_coord] *= factor.p[factor_coord]
      log_joint_acc += np.log(factor_acc)
    log_joint_acc -= np.max(log_joint_acc) # to avoid numerical issues
    joint_acc = np.exp(log_joint_acc) / np.sum(np.exp(log_joint_acc))
    # compute marginals
    for i, variable in enumerate(variables):
      sum_dims = [j for j in range(N) if not j == i]
      sum_dims.sort(reverse=True)
      collapsing_marginal = joint_acc
      for j in sum_dims:
        collapsing_marginal = collapsing_marginal.sum(j) # lose 1 dim
      variable.bfmarginal = collapsing_marginal
    return variables
