
from stellacode.optimizer import Optimizer

def test_optimization():
    path_config_file = "test/data/li383/config.ini"

    opt = Optimizer.from_config_file(path_config_file)
    opt.max_iter = 1
    opt.method="L-BFGS-B"
    opt.options = {"disp": True, "maxls": 1}
    opt.optimize()
