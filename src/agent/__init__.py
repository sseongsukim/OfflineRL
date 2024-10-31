from src.agent import cql
from src.agent import iql
from src.agent import td3bc

model_free_algos = {
    "iql": (iql.create_learner, iql.get_default_config()),
    "cql": (cql.create_learner, cql.get_default_config()),
    "td3bc": (td3bc.create_learner, td3bc.get_default_config()),
}