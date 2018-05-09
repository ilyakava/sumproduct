import unittest
import numpy as np

from sumproduct import Variable, Factor, FactorGraph, Mu


class SimpleGraph(unittest.TestCase):
    """
  This is the graph pictured in the readme.

  Learn more about this graphical model from exercise 8.10
  and figure 8.54 in Christopher Bishop's Pattern Recognition
  and Machine Learning
  """

    def createSimpleGraph(self):
        # create the orphaned nodes
        node_names = ['x1', 'x2', 'x3', 'x4']
        dims = [2, 3, 2, 2]

        # pad array just so that reference like x[1] later is easier to read
        x = [None] + [Variable(node_names[i], dims[i]) for i in range(4)]

        f3 = Factor('f3', np.array([0.2, 0.8]))
        f4 = Factor('f4', np.array([0.5, 0.5]))

        # first index is x3, second index is x4, third index is x2
        # looking at it like: arr[0][0][0]
        f234 = Factor('f234', np.array([
            [
                [0.3, 0.5, 0.2], [0.1, 0.1, 0.8]
            ], [
                [0.9, 0.05, 0.05], [0.2, 0.7, 0.1]
            ]
        ]))

        # first index is x2
        f12 = Factor('f12', np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]))

        # attach nodes to graph in right order (connections matching
        # factor's potential's dimensions order)
        g = FactorGraph(x[3], silent=True)
        g.append('x3', f234)
        g.append('f234', x[4])
        g.append('f234', x[2])
        g.append('x2', f12)
        g.append('f12', x[1])
        g.append('x3', f3)
        g.append('x4', f4)
        return g

    def setUp(self):
        self.g = self.createSimpleGraph()

    def testTwoIndependentInstances(self):
        g1 = self.createSimpleGraph()
        g2 = FactorGraph()
        self.assertTrue(len(g1.nodes))
        self.assertTrue(len(g2.nodes) == 0)

    def testCustomErrorFunction(self):
        def func(m1, m2):
            return sum([sum(np.absolute(m1[k] - m2[k])) for k in m1.keys()])

        self.g.compute_marginals(error_fun=func)

    def testSumProductInference(self):
        self.g.compute_marginals()
        self.assertTrue(
            np.allclose(self.g.nodes['x1'].marginal(), np.array([0.536, 0.464
                                                                 ])))
        self.assertTrue(
            np.allclose(self.g.nodes['x2'].marginal(), np.array([0.48, 0.36,
                                                                 0.16])))
        self.assertTrue(
            np.allclose(self.g.nodes['x3'].marginal(), np.array([0.2, 0.8])))
        self.assertTrue(
            np.allclose(self.g.nodes['x4'].marginal(), np.array([0.5, 0.5])))

    def testBruteForceInference(self):
        self.g.brute_force()
        self.assertTrue(
            np.allclose(self.g.nodes['x1'].bfmarginal, np.array([0.536, 0.464
                                                                 ])))
        self.assertTrue(
            np.allclose(self.g.nodes['x2'].bfmarginal, np.array([0.48, 0.36,
                                                                 0.16])))
        self.assertTrue(
            np.allclose(self.g.nodes['x3'].bfmarginal, np.array([0.2, 0.8])))
        self.assertTrue(
            np.allclose(self.g.nodes['x4'].bfmarginal, np.array([0.5, 0.5])))


class InboxToMarginal(unittest.TestCase):
    def setUp(self):
        node = Variable('bit', 2)
        uniform = Mu(None, np.array([0.5, 0.5]))
        point = Mu(None, np.array([1.0, 0.0]))
        node.deliver(2, uniform)
        node.deliver(2, point)
        self.node = node

    def testFewHarshProbabilities(self):
        self.assertTrue(
            np.allclose(self.node.marginal(), np.array([1.0, 0.0])))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
