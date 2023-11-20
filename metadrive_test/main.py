import random
import matplotlib.pyplot as plt
from metadrive.component.map.base_map import BaseMap
from metadrive import TopDownMetaDrive,TopDownMetaDriveEnvV2,MetaDriveEnv
from metadrive.constants import HELP_MESSAGE
from metadrive.examples.ppo_expert.numpy_expert import expert


if __name__ == "__main__":
    env = MetaDriveEnv(
        dict(
            map="SSSSSSSSSSS",
            traffic_density=0.1,
            num_scenarios=100,
            use_render = True,
            start_seed=random.randint(0, 1000),
            # manual_control= True,
            show_coordinates=True,
            # show_policy_mark=True,
            image_observation=False,
            vehicle_config=dict(
                lidar=dict(
                    add_others_navi=False,
                    num_others=4,
                    distance=100,
                    num_lasers=1,
                )
            )
        )
    )
    try:
        o, _ = env.reset()
        for i in range(1, 100000):
            # o, r, tm, tc, info = env.step(expert(env.vehicle))
            o, r, tm, tc, info = env.step([0,0.1])
            # print(env._wrap_as_single_agent(env.get_single_observation()))
            # print(o.shape)
            # print(o.shape)
            env.render(mode="human",film_size=(1000, 1000))#human,rgb_array,["top_down", "topdown", "bev", "birdview"]
            if tm or tc:
                env.reset()
    except Exception as e:
        raise e
    finally:
        env.close()