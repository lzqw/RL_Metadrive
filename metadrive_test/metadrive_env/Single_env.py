import random
import matplotlib.pyplot as plt
from metadrive.component.map.base_map import BaseMap
from metadrive import TopDownMetaDrive,TopDownMetaDriveEnvV2
from metadrive.constants import HELP_MESSAGE
from metadrive.examples.ppo_expert.numpy_expert import expert



if __name__ == "__main__":
    env = TopDownMetaDrive(
        dict(
            map="SSS",
            traffic_density=0.2,
            num_scenarios=100,
            # use_render = True,
            start_seed=random.randint(0, 1000),

        )
    )
    try:
        o, _ = env.reset()
        for i in range(1, 100000):
            # o, r, tm, tc, info = env.step(expert(env.vehicle))
            o, r, tm, tc, info = env.step([0,0.1])
            env.render(mode="topdown",film_size=(1000, 1000))#human,rgb_array,["top_down", "topdown", "bev", "birdview"]
            if tm or tc:
                env.reset()
    except Exception as e:
        raise e
    finally:
        env.close()