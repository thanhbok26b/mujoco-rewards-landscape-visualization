import os
import yaml

BASEDIR = os.path.dirname(__file__)

config = yaml.load(open(os.path.join(BASEDIR, 'config.yaml')).read())

CAPACITY        = config['redis_keys']['capacity']
ENV_NAMES       = config['redis_keys']['env_names']
POLICY          = config['redis_keys']['policy']
NORMALIZER      = config['redis_keys']['normalizer']
NORMALIZER_HASH = config['redis_keys']['normalizer_hash']
JOB             = config['redis_keys']['job']
TRAJECTORY      = config['redis_keys']['trajectory']
ENV_NAME        = config['redis_keys']['env_name']

benchmarks      = config['benchmarks']