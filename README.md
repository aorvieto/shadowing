# Shadowing Properties of Optimization Algorithms
## by Antonio Orvieto and Aurelien Lucchi

Paper: http://people.inf.ethz.ch/orvietoa/shadowing.pdf

Ordinary differential equation (ODE) models of gradient-based optimization methods can provide insights into the dynamics of learning and inspire the design of new algorithms. Unfortunately, this thought-provoking perspective is weakened by the fact that, in the worst case, the error between the algorithm steps and its ODE approximation grows exponentially with the number of iterations. In an attempt to encourage the use of continuous-time methods in optimization, we show that, if some additional regularity on the objective is assumed, the ODE representations of Gradient Descent and Heavy-ball do not suffer from the aforementioned problem, once we allow for a small perturbation on the algorithm initial condition. In the dynamical systems literature, this phenomenon is called shadowing. Our analysis relies on the concept of hyperbolicity, as well as on tools from numerical analysis.

