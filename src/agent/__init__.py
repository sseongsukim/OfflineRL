from src.agent import cql, iql, td3bc
from src.agent import mopo, rambo, mobile


model_free_algos = {
    "iql": (iql.create_learner, iql.get_default_config()),
    "cql": (cql.create_learner, cql.get_default_config()),
    "td3bc": (td3bc.create_learner, td3bc.get_default_config()),
}

model_based_algos = {
    "mopo": (mopo.create_learner, mopo.get_default_config()),
    "rambo": (rambo.create_learner, rambo.get_default_config()),
    'mobile': (mobile.create_learner, mobile.get_default_config()),
}