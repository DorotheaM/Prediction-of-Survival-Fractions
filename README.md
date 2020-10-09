# Survival of human cancer cells after their exposure to ionizing radiation
Here, you will find the code for my thesis project. It is written in Python.
The aim was to develop a predictive model that would study the effects of ionizing radiation (X and γ rays) on human cancer cells with the use of ML algorithms.
The models are based on the bagging algorithm, Random Forest, and the boosting algorithm, Gradient Boosting.
The pipeline is as follows: 
First, we study the base model using the default parameters provided by scikit-learn.
Next, we create a grid in order to find the parameters that optimize each model.
Finally, we use said parameters to train again our model and evaluate its potential improvement.
