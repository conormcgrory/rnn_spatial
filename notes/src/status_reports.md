# Status reports

## 2022-02-08

- Still not sure how Cueva and Wei are creating simulated animal trajectories
- Network structure and regularization is more clear
- Goal for this week: Finish network
    - Want to get network to a point where it is successfully performing path integration on our current rough trajectory samples
	- Will work on trajectories, including possibility of using run-and-tumble models, after network is up and running

## 2022-02-16

- Have simple, "hacky" trajectory simulation up and running
- Network was harder to get working than we originally thought
- Trying to figure out how to train Cueva and Wei network -- they use Hessian-free optimization, which is probably too difficult to implement ourselves
- Goal for this week: Get _some_ network running on trajectory data, even if it's simpler than Cueva and Wei model

## 2022-02-22

- Skipped meeting with Braden this week because I didn't have anything new to show
- Coded up network model in PyTorch today; got it to train
- Goal for this week: Create visualization tools for evaluating network performance
	- Need to do some refactoring in simulation code; want to create `Simulation` object that stores batch of trials, parameters
	- Also need to create plot of predicted trajectories vs. ground truth

## 2022-03-02

- Network is not successfully learning to perform path integration
- Talked with Braden; we thought of a number of different things to try to get network to function
- Goal for this week: Get network to learn path integration on simple simulation data
