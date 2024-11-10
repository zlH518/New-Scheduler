import logging
import random

def setup_seed(seed):
    random.seed(seed)


def setup_logging(config):
    logging.basicConfig(
        filename=config["log_file"],
        format=config["format"],
        level=config["log_level"],
        filemode=config["filemode"]
    )
if __name__ == "__main__":
    import yaml
    with open('code/example/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    setup_logging(config["logger"])
    logging.info("hello")
    