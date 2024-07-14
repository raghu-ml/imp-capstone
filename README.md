# 2024 imerial AI/ML capstone

## Summary
This project is my submission for the Blackbox optimization capstone competition for 2024 Imperial College Professional Certificate in Machine Learning.

The project is a competition that requires optimization of 8 blackbox functions. We use a Bayesian optimization to solve this problem.
The code here deomstrate the solution. We will also document reflection and approaches taken.

## Inital data
We were provided with  inital observation. The directory `initial_data` contains the inital data for the submission.

## Feedback on suggestions
Subsequent feedback is in `588_data.csv`

## Running 
The optimization can be run by running the tests in `tests/test_Optimize.py`.
This will read the initial data and join this with suggestions and run the optimizer and output next suggestion.

Below is an example of the the rest run looks like. The red box outlines the suggestion for a function, correctly formatted. Note that there is one test for each function.

![test_output](https://github.com/user-attachments/assets/8e5056d6-6b47-490c-a165-bb5faebb0cb5)


## Approach and Code structure

My aim was to implement the `sklearn.GaussianProcessRegressor` as we did the lecture notes. Infact, I copy that implementation as is. However, given that there are 8 functions to optimize, I wanted an implementation that could be easily customized, specifically, I wanted to be able to try different covariance kernels and acquisition functions. I also wanted parameters to control exploration vs exploitation and finally I wanted to optimize the acquisition functions by not just a random state space search. Instead i wanted to employ some optimization technique.

Firstly for optimization i copied the code from https://github.com/bayesian-optimization/BayesianOptimization/tree/master/bayes_opt.

From here we only take the `bayes_opt/util.py` package. This package contain implementation of Upper Confidence bound, Expected improvement and Probability of Improvement.

I reviewed the implementation with these references: 
* https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf
* https://medium.com/@okanyenigun/step-by-step-guide-to-bayesian-optimization-a-python-based-approach-3558985c6818

This package also contains a function `acq_max` which can maximise the acquisition fucnction for the given Gaussian process. This randomly samples the the function `n_warmup` times and then uses the scipy optimize package with `n_iter` random seeds. We take the best guess.

Finally we put all of this together, for each function (1 to 8), we create a Gaussian process regressor with the following parameters:
* Covariance kernel - with kernel parameters
* Acquisition function - ucb (upper confidence bound), poi (probability of improvement), ei (expected imorovement)
* bounds - focus the search for suggestion within these bounds - useful when we want exploit in a certain area.


### The approach

I started with the approach of running the optimization for each function using the UCB method with a kappa of 2 to force a high degree if exploration. The default kernel was the Marten kernel. I noticed that there were convergence issues for function 1 and 5. I experimented with different kernels and settled on the radial basis kernel. After a few weeks of using ucb I moved to poi method and still keeping the exploration parameter positive. I set different exploration parameters for different functions, prefering the higher values for the higher dimensional functions. Expected improvement acquisition seemed to get stuck in optimization phase, collapsing to previously seen values.

After a few more observations (as we got close to sumbission, with 2 submissions per week), i removed the exploration paramteter prefering exploitation.

I also set the search bounds on function 1 to `[[0.77,0.8],[0.64,0.75]]` as i felt the suggestions were outside the range where I had observed the maximum values.

I also explored different kernels and did not notice any differene in the suggestions and decided to stick with the default Marten kernel.

I feel like I have increased the number of iterations of the optimizer at the end, to maximize the exploitation, we went from 10 to 1000.

