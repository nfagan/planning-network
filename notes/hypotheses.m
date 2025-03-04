%{

Does the slope of performance per additional tick of recurrent processing 
show a similar trajectory to the use of planning over learning, and as a
function of network capacity?

todo:

(0) Save evaluation results as .mat files for analysis in matlab

1. for each model checkpoint, compute performance for a fixed number of
   ticks applied at the start of the explore phase.

2. for each checkpoint, fit log/exp models of performance as a function of 
   ticks.

3. plot slopes as a trajectory over checkpoints, i.e., over the course of 
   learning, separately for small vs. large networks.

%}