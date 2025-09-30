# FPSNN
Code for "Physics-informed Spiking Neural Networks for Continuous-Time Dynamic Systems".

## Contents
```
FPSNN/
  data/                   
    # error and loss data
  model_params/           
    # includes one example for each model to compute energy consumption
  packs/                  
    # models, equations, etc.
  energy_consumption.py
    # graphs energy consumption using FPSNN/model_params/
  execute_params.py
    # hyperparameters
  graph_error_loss.py   
    # graphs error and loss curves from data in FPSNN/data/
  graph_res.py
    # plots reference and model solutions and errors
  graph_res_2d.py
    # plots reference and model solutions and errors (Burger's 2D)
  graph_surrogate.py
    # plots surrogate gradients
  main.py
    # training models
  main_burgers2d.py
    # training models (Burger's 2D)
```
