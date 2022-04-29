# CS-594 Project
___
# Reinforcement Learning for Financial Stock Trading Applications

### Project by: Rajat Kumar and Sachin Parsa

This project contains the code for running modified as well as Ensemble strategy for Dow & Jones 30 Stock trading.

Algorithms Used:
> A2C (Advantage Actor Critic)

> PPO (Proximal Policy Optimization)

> DDPG (Deep Deterministic Policy Gradient)

> TD3 (Twin Delay DDPG)


### How to run?
___
First clone this repo. Then open the project in an IDE (like VS Code) or a Jupyter Notebook.

Code for running original ensemble strategy:

Default - 3 month validation and model selection window:
```
python Ensemble/run_original_ensemble.py

# or 

python Ensemble/run_original_ensemble.py three
```

1 month validation and model selection window:
```
python Ensemble/run_original_ensemble.py one
```

Half month validation and model selection window:
```
python Ensemble/run_original_ensemble.py half
```

Code for running modified ensemble strategy:

Default - 3 month validation and model selection window:
```
python Ensemble/run_modified_ensemble.py

# or

python Ensemble/run_modified_ensemble.py three
```

1 month validation and model selection window:
```
python Ensemble/run_modified_ensemble.py one
```

Half month validation and model selection window:
```
python Ensemble/run_modified_ensemble.py half
```

Dynamic model selection with validation window
```
# 3 month/default
python Ensemble/run_modified_ensemble.py three True

# one month
python Ensemble/run_modified_ensemble.py one True

# half month
python Ensemble/run_modified_ensemble.py half True
```

You have have a look at the captured results in the notebooks here:
https://github.com/RJonMshka/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/blob/master/ensemble_original_1_month_window.ipynb

https://github.com/RJonMshka/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/blob/master/ensemble_original_3_month_window.ipynb

https://github.com/RJonMshka/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/blob/master/ensemble_original_half_month_window.ipynb

https://github.com/RJonMshka/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/blob/master/ensemble_td3_1_month.ipynb

https://github.com/RJonMshka/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/blob/master/ensemble_td3_3_month.ipynb

https://github.com/RJonMshka/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/blob/master/ensemble_td3_half_month.ipynb

### Results captured for the experiment are stored in CSV files here:
https://github.com/RJonMshka/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/tree/master/Ensemble%20Results



Thanks
Rajat and Sachin