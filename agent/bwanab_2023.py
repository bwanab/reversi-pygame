# bwanab_2023

from agent.base_agent import BaseAgent
import gym
import boardgame2
from tictactoe.ttt_az_torch import *
from util.util import simple_heuristic_action, position_based_heuristic_action, random_action
from util.util import position_based_heuristic_action_enhanced, position_plus_min_liberty_action, hybrid_action
from util.util import reversi_ai_action
import numpy as np
import pygame


class BBAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = 1 if kwargs['color'] == 'white' else -1
        self.gym_env = gym.make("Reversi-v0", board_shape=8)
        self.num_zeros = 0
        self.agent = AlphaZeroAgent(env=self.gym_env, model_name="bwanab_2023/reversi8_512", is_train=False, is_playing_human=True)

    def step(self, reward, obs):
        n_zeros = sum([0 == x for x in obs.values()])
        if n_zeros > self.num_zeros:
            self.agent.reset_mcts()
        self.num_zeros = n_zeros
        o = np.array([obs[x] for x in range(len(obs))]).reshape((8,8))
        observation = (o, self.player)
        action = self.agent.step(observation, 0, False)
        return (self.col_offset + action[1] * self.block_len, self.row_offset + action[0] * self.block_len), pygame.USEREVENT

class SimpleHeuristicAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = 1 if kwargs['color'] == 'white' else -1

    def step(self, reward, obs):
        o = np.array([obs[x] for x in range(len(obs))]).reshape((8,8))
        observation = (o, self.player)
        x, y = simple_heuristic_action(observation)
        return (self.col_offset + y * self.block_len, self.row_offset + x * self.block_len), pygame.USEREVENT


class PositionHeuristicAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = 1 if kwargs['color'] == 'white' else -1

    def step(self, reward, obs):
        o = np.array([obs[x] for x in range(len(obs))]).reshape((8,8))
        observation = (o, self.player)
        x, y = position_based_heuristic_action(observation)
        return (self.col_offset + y * self.block_len, self.row_offset + x * self.block_len), pygame.USEREVENT

class PositionHeuristicAgentEnhanced(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = 1 if kwargs['color'] == 'white' else -1

    def step(self, reward, obs):
        o = np.array([obs[x] for x in range(len(obs))]).reshape((8,8))
        observation = (o, self.player)
        x, y = position_based_heuristic_action_enhanced(observation)
        return (self.col_offset + y * self.block_len, self.row_offset + x * self.block_len), pygame.USEREVENT

class PositionHeuristicAgentPlusMinMoves(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = 1 if kwargs['color'] == 'white' else -1

    def step(self, reward, obs):
        o = np.array([obs[x] for x in range(len(obs))]).reshape((8,8))
        observation = (o, self.player)
        x, y = position_plus_min_liberty_action(observation)
        return (self.col_offset + y * self.block_len, self.row_offset + x * self.block_len), pygame.USEREVENT

class RandomAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = 1 if kwargs['color'] == 'white' else -1

    def step(self, reward, obs):
        o = np.array([obs[x] for x in range(len(obs))]).reshape((8,8))
        observation = (o, self.player)
        x, y = random_action(observation)
        return (self.col_offset + y * self.block_len, self.row_offset + x * self.block_len), pygame.USEREVENT

class HybridAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = 1 if kwargs['color'] == 'white' else -1

    def step(self, reward, obs):
        o = np.array([obs[x] for x in range(len(obs))]).reshape((8,8))
        observation = (o, self.player)
        x, y = hybrid_action(observation, 10)
        return (self.col_offset + y * self.block_len, self.row_offset + x * self.block_len), pygame.USEREVENT

from reversi_ai.reversi import *
from reversi_ai.aihelper import FormatConverter
from reversi_ai.reversiai import *

class Reversi_AI_Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = 1 if kwargs['color'] == 'white' else -1

        # self.rai = ReversiAI()
        # self.r_ai_player = 'w' if self.player == 1 else 'b'

    def step(self, reward, obs):
        o = np.array([obs[x] for x in range(len(obs))]).reshape((8,8))
        observation = (o, self.player)
        x, y = reversi_ai_action(observation)


        # cell_map = {1: 'w', 0: ' ', -1: 'b'}
        # rai_board = []
        # for x in range(8):
        #     rai_board.append([])
        #     for y in range(8):
        #         rai_board[x].append(cell_map[o[x, y]])

        # try:
        #     coord = self.rai.get_next_move(rai_board, self.r_ai_player)
        #     x = coord.x
        #     y = coord.y
        # except GameHasEndedError:
        #     o = np.array([obs[x] for x in range(len(obs))]).reshape((8,8))
        #     observation = (o, self.player)
        #     x, y = random_action(observation)
        return (self.col_offset + y * self.block_len, self.row_offset + x * self.block_len), pygame.USEREVENT
