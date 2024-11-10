import yaml
import utils
from Executor import Executor
from plot import Plot
from datetime import datetime


if __name__ == "__main__":
    args = utils.get_cmd_args()
    utils.setup_seed(args.seed)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    utils.setup_logging(config["logger"])
    executor = Executor(config)
    executor.executor()
    Plot.plot(config)


