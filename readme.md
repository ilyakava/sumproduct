# sumproduct

An implementation of Belief Propagation for factor graphs, also known as the sum product algorithm.

![Simple factor graph](http://f.cl.ly/items/2P021j2y3A2Q191F451h/unnamed0.png)

The factor graph used in `test.py`.

## Usage

Check `test.py` for details, but:

### Create a factor graph

```
g = FactorGraph() # init the graph
x1 = Variable('x1', 2) # init a variable with 2 states
x2 = Variable('x2', 3) # init a variable with 3 states
f12 = Factor('f12', np.array([
  [0.8,0.2],
  [0.2,0.8],
  [0.5,0.5]
])) # create a factor node potentials for p(x1 | x2)
# connect the parents to their children
g.add(f12)
g.append('f12', x2)
g.append('f12', x1)
```

### Run Inference

#### sum-product algorithm

```
g.compute_marginals(max_iter=500, tolerance=1e-6)
g.nodes['x1'].marginal()
```

#### Brute force marginalization and conditioning

The sum-product algorithm can only compute exact marginals for acyclic graphs. Check against the brute force method (at great computational expense) if you have a loopy graph.

```
res = g.brute_force()
g.nodes['x1'].bfmarginal
```

## Implementation Details

See block comments in the code's methods for details, but implementation strategy comes from Chapter 5 of [David Barber's book](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.HomePage).
