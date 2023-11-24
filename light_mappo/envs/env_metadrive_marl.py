import numpy as np

sub_agent_obs = []
for i in range(2):
    sub_obs = np.random.random(size=(14,))
    sub_agent_obs.append(sub_obs)



sub_agent_obs = []
sub_agent_reward = []
sub_agent_done = []
sub_agent_info = []

for i in range(2):
    sub_agent_obs.append(np.random.random(size=(14,)))
    sub_agent_reward.append([np.random.rand()])
    sub_agent_done.append(False)
    sub_agent_info.append({})
print(sub_agent_done)
print(sub_agent_info)