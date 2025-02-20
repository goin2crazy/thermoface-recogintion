import yaml

def load_config(path = "config.yaml"): 

    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)