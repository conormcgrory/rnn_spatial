# Initial project goals

- _Goal_: Reproduce some of the results of Cueva and Wei, 2018
- Roadmap:
      1. Write code for simulating animal motion
      2. Train RNN on simulated trajectories 
      3. Examine RNN units to see if any have grid, place, or border cell properties
- Motion simulation
	- Current status: direction is sampled from Brownian motion process, and speed is sampled from categorical distribution
	- Need to try other approaches once network is training successfully
- Network training
	- Going to start with simple Elman RNN trained with Adam
	- Might need to try Hessian-free optimization algorithm 
