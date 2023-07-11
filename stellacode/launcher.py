import logging
import sys

import jax

from stellacode.optimizer import Optimizer

logger = logging.getLogger(__name__)
jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    opt = Optimizer.from_config_file(sys.argv[1])
    opt.optimize()
