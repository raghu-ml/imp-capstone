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
