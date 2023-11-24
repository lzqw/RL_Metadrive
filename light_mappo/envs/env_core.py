import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv,MultiAgentStraightEnv
)
import argparse
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.idm_policy import ManualControllableIDMPolicy

envs = dict(
    roundabout=MultiAgentRoundaboutEnv,
    intersection=MultiAgentIntersectionEnv,
    tollgate=MultiAgentTollgateEnv,
    bottleneck=MultiAgentBottleneckEnv,
    parkinglot=MultiAgentParkingLotEnv,
    pgma=MultiAgentMetaDrive,
    straight=MultiAgentStraightEnv
)

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self,args):
        self.args=args
        env_cls_name = args.env
        self.env= envs[env_cls_name](
        {
            "use_render": True if not args.top_down else False,
            "crash_done": True,
            "sensors": dict(rgb_camera=(RGBCamera, 512, 256)),
            "interface_panel": ["rgb_camera", "dashboard"],
            # "agent_policy": ManualControllableIDMPolicy,
            "num_agents" :self.args.num_agents,
        }
         )
        self.agent_num = self.args.num_agents
        self.obs_dim = list(self.env.observation_space.values())[0].shape[0]
        self.action_dim = list(self.env.action_space.values())[0].shape[0]

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        state = self.env.reset()
        sub_agent_obs=list(state[0].values())

        """
        [array([0.49429161, 0.11767061, 0.08365448, 0.80148316, 0.00954728,
       0.17152566, 0.90665392, 0.16911162, 0.29806652, 0.25131156,
       0.99582807, 0.61365704, 0.82655726, 0.16925658]), 
       array([0.98333378, 0.96975691, 0.56973207, 0.91186993, 0.20662234,
       0.69301029, 0.93051759, 0.79534628, 0.06723421, 0.25382497,
       0.19824206, 0.65885847, 0.62897183, 0.28868158])]
        """
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        sub_agent_obs,reward,done,truncateds,info = self.env.step({agent_id: action for agent_id,action in zip(self.env.vehicles.keys(),actions)})
        sub_agent_obs=list(sub_agent_obs.values())
        sub_agent_reward=list(reward.values())
        sub_agent_done = list(done.values())[:-1]
        sub_agent_info = list(info.values())
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="straight", choices=list(envs.keys()))
    parser.add_argument("--top_down", action="store_true")
    parser.add_argument("--num_agents", type=int,default=2)
    args = parser.parse_args()
    my_env=EnvCore(args)
    my_env.reset()
    action=[np.zeros(2),np.zeros(2)]
    my_env.step(action)