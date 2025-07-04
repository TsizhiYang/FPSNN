network_type = "snn"            # "snn", "ann"
equation_type = "heat"          # "heat", "wave"
task_type = "dl"               # "fdm", "std", "dl"

initialization = "xavier"       # "undef", "xavier"


if equation_type == "heat":
    trials = 10
    epoches = 200000
    record_epoch = 1000
    show_epoch = 10000
    graph_epoch = 10000
    learning_rate = 1e-4
    phys_weight = 1
    cond_weight = 500000
elif equation_type == "wave":
    trials = 10
    epoches = 250000
    record_epoch = 1000
    show_epoch = 10000
    graph_epoch = 10000
    learning_rate = 1e-3
    phys_weight = 10
    cond_weight = 10000

show_zero_epoch_graph = True
t_axis_lines = 15


if network_type == "snn" and task_type == "fdm":
    label = "SNN FDM-PINN"
    name = "SNN_FDM"
if network_type == "snn" and task_type == "std":
    label = "SNN PINN"
    name = "SNN_STD"
if network_type == "snn" and task_type == "dl":
    label = "SNN DD"
    name = "SNN_DL"
if network_type == "ann" and task_type == "fdm":
    label = "ANN FDM-PINN"
    name = "ANN_FDM"
if network_type == "ann" and task_type == "std":
    label = "ANN PINN"
    name = "ANN_STD"
if network_type == "ann" and task_type == "dl":
    label = "ANN DD"
    name = "ANN_DL"
