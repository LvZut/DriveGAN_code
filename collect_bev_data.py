# File: collect_bev_data.py
# Desc: Data generator for Drivegan
# Auth: Louis van Zutphen
#
# Copyright: Saivvy 2021
#
##################################################


import gym
import carla_gym
import numpy as np
from numpy import random
from pathlib import Path
from PIL import Image
import json
import os

class random_policy():
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space


    def step(self):
        # take random action
        while True:
            action = self.action_space.sample()
            if action[0] > 0:
                break
        obs, r, done, info = self.env.step(action)

        return (obs, r, done, info), action

def main():
    # get save path
    os.umask(000)
    episodes_folder = 'data3'
    PATH = f'latent_decoder_model/img_data/{episodes_folder}/'
    # os.makedirs(PATH, 0o777)
    Path(PATH).mkdir(parents=True, exist_ok=True)
    

    # init GYM

    # sensors
    bev = carla_gym.sensors.BirdeyeSensor(params={"width": 256, "height": 256, "show_window": False, "bev_type": 'RGB', 'pixels_per_meter' : 6})
    params = {"sensors": [bev], 'max_episode_length': 80, 'carla_host': 'localhost'}
    # other params

    env = gym.make('carla-v0', params=params)

    # policy
    policy = random_policy(env)
    

    # 3000 episodes of 80 images each
    episodeLength = 80
    maxEpisodes = 300000






    # get data from policy using gym

    episode = 0
    while episode < maxEpisodes:
        percentage = round(float(episode)/float(maxEpisodes), 3)
        print(f'Episode: {episode}/{maxEpisodes} ({percentage})')
        # make directory for episode
        episode_PATH = PATH+str(episode)+'/'
        Path(episode_PATH).mkdir(parents=True, exist_ok=True)

        
        obs = env.reset()
        # save obs

        episode_dict = {    "extra": {
                            "init_ix": 1526, # start position?
                            "ego_ix": 22, # car model
                            "weather_ix": 7, # weather settings
                            "nnpc_ix": 144 }, # number of npcs?
                            "data" : []
                        }

        # get images for episode
        step = 0
        while step < episodeLength:
            # policy step

            (obs, _, _, _), action = policy.step()

            # breakpoint()
            # save obs images
            im = Image.fromarray(obs[0])
            im.save(f"{episode_PATH}{step}.png")

            ego = env.ego
            ego_av = ego.get_angular_velocity()

            # breakpoint()
            # add JSON entry
            entry = {
                    "speed": ego.get_velocity().length(),
                    "steer": action[1].item(),
                    "throttle": action[0].item(),
                    "angular_velocity": [ego_av.x, ego_av.y, ego_av.z]  # must be list, not sure if this works
                    }
            episode_dict['data'].append(entry)
            
            step += 1
        
        # save JSON
        with open(f'{episode_PATH}info.json', 'w') as fp:
            json.dump(episode_dict, fp)
        # break
        episode += 1



if __name__ == '__main__':
    main()
