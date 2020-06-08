import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from time import sleep

class agentAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(15*30,12) ## THis has to altered for image size
        ## Output has to be 12 with respect to joints
        self.Fp = nn.Sequential(
                    nn.Conv2d(3,10,5,stride=3),
                    nn.BatchNorm2d(10)
                    nn.ReLU(),
                    nn.MaxPool2d(3,3),                    
                    )
    def forward(self,inputs):
        x = self.Fp(inputs)
        x = x.reshape(x.size(0),-1)
        x = self.fc1(x)
        return x

def init_weights(m):
    if ((type(m) == nn.Conv2d) or (type(m) == nn.Linear)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)

def create_agents(n): ##For evolutionary
    agents = []
    for i in range(n):
        agent=agentAI()

        for param in agent.parameters():
            param.requires_grad = False
        init_weights(agent)

        agents.append(agent)
    return agents
def fit(agents,envi,human=False,sti=1):
    return_agents[]
    for agent in agents:  ## USe Thread to finish the process faster
        agent.eval()
        observation = envi.reset()
        rew=0
        while True:
            if human:
                time.sleep(sti)
            observation = torch.tensor(observation)
            inp = observation.type('torch.FloatTensor').view(-1,1)
            action = agent(inp).detach().numpy()[0] ## This is gonna be list of 12 no of  
            for i in range(len(action)):
                action[i]*=envi.action_space.high[i]
            observation,reward,done,info = envi.step(action)
            rew =rew+reward
            if (done):
                break
        return_agents.append(rew)
    return return_agents

def run_agents(agents):
    envs = []
    for i in range(noc):
        envs.append(env)
    env.reset()

    agents = np.array(agents)
    agents = agents.reshape(noc,-1)
    result_id=[]
    for i in range(noc):
        result_id.append(fit(agents[i],envs[i]))

    results = result_id
    results = np.array(results,dtype=int)
    return results.reshape(agents.shape[0]*agents.shape[1],-1)


