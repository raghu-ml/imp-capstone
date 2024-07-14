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
My approach this capstone was a simple one.

My aim was to implement the `sklearn.GaussianProcessRegressor` as we did the lecture notes. Infact, I copy that implementation as is. However, given that there are 8 functions to optimize, I wanted an implementation that could be easily customized, specifically, I wanted to be able to try different covariance kernels and acquisition functions. I also wanted parameters to control exploration vs exploitation and finally I wanted to optimize the acquisition functions by not just a random state space search. Instead i wanted to employ some optimization technique.

Firstly for optimization i copied the code from https://github.com/bayesian-optimization/BayesianOptimization/tree/master/bayes_opt.

From here we only take the `bayes_opt/util.py` package. This package contain implementation of Upper Confidence bound, Expected improvement and Probability of Improvement.

I reviewed the implementation with these references: h
* https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf
* https://medium.com/@okanyenigun/step-by-step-guide-to-bayesian-optimization-a-python-based-approach-3558985c6818

  


