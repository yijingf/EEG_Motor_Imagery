# 19082919

{"input": {"step": 10, "window_len": 20}, "rnn": {"bidirect": false, "n_hidden": [64], "weight_decay": 0.0001}, "cnn": {"fc_n_hidden": 1024, "net_type": "CNN", "conv_filter_shape": [{"W": [3, 3, 1, 32], "b": [32]}, {"W": [3, 3, 32, 64], "b": [64]}, {"W": [3, 3, 64, 128], "b": [128]}], "fc_layer": true, "reuse": true, "weight_decay": 0.0005}}


# 19082923

{"cnn": {"conv_filter_shape": [{"W": [3, 3, 1, 32], "b": [32]}, {"W": [3, 3, 32, 64], "b": [64]}, {"W": [3, 3, 64, 128], "b": [128]}], "net_type": "CNN", "reuse": true, "fc_n_hidden": 1024, "fc_layer": true, "weight_decay": 0.0005}, "rnn": {"n_hidden": [64], "weight_decay": 0.0001, "bidirect": false}, "input": {"step": 10, "window_len": 20}}


# 20190828

{"input": {"step": 10, "window_len": 20}, "cnn": {"net_type": "CNN", "fc_n_hidden": 1024, "fc_layer": true, "weight_decay": 0.0005, "conv_filter_shape": [{"b": [32], "W": [3, 3, 1, 32]}, {"b": [64], "W": [3, 3, 32, 64]}, {"b": [128], "W": [3, 3, 64, 128]}], "reuse": true}, "rnn": {"weight_decay": 0.0001, "n_hidden": [64, 16], "bidirect": false}}


# 20190829

{"rnn": {"weight_decay": 0.0001, "bidirect": true, "n_hidden": [64]}, "input": {"window_len": 20, "step": 10}, "cnn": {"weight_decay": 0.0005, "fc_layer": true, "fc_n_hidden": 1024, "reuse": true, "net_type": "CNN", "conv_filter_shape": [{"b": [32], "W": [3, 3, 1, 32]}, {"b": [64], "W": [3, 3, 32, 64]}, {"b": [128], "W": [3, 3, 64, 128]}]}}
