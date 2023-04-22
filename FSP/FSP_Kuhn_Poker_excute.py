#Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
from collections import defaultdict
import sys
from tqdm import tqdm
import time
import doctest
import copy
from sklearn.neural_network import MLPClassifier
from collections import deque
import wandb
import random
import FSP_Kuhn_Poker_trainer

nash_equilibrium = {'J': np.array([0.8, 0.2]), 'Jb': np.array([1,0]) , 'Jp': np.array([0.67, 0.33]), 'Jpb': np.array([1, 0]), 
                    'K': np.array([0.25, 0.75]), 'Kb': np.array([0,1]), 'Kp': np.array([0,1]), 'Kpb': np.array([0,1]),
                      'Q': np.array([1,0]), 'Qb': np.array([0.67, 0.33]), 'Qp': np.array([1,0]), 'Qpb': np.array([0.4, 0.6])}

class KuhnTrainer_XFP:
  def __init__(self, train_iterations=10**4):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = 2
    self.PASS = 0
    self.BET = 1
    self.NUM_ACTIONS = 2
    self.avg_strategy = {}
    self.rank = {"J":1, "Q":2, "K":3}
    self.differences = {}

  #return util for terminal state
  def Return_payoff_for_terminal_states(self, history, target_player_i):

    plays = len(history)
    player = plays % 2
    opponent = 1 - player
    terminal_utility = 0

    if plays > 3:
      terminalPass = (history[plays-1] == "p")
      doubleBet = (history[plays-2 : plays] == "bb")
      isPlayerCardHigher = (self.rank[history[player]] > self.rank[history[opponent]])

      if terminalPass:
        if history[-2:] == "pp":
          if isPlayerCardHigher:
            terminal_utility = 1
          else:
            terminal_utility = -1
        else:
            terminal_utility = 1
      elif doubleBet: #bb
          if isPlayerCardHigher:
            terminal_utility = 2
          else:
            terminal_utility = -2

    if player == target_player_i:
      return terminal_utility
    else:
      return -terminal_utility



  #whether terminal state
  def whether_terminal_states(self, history):

    plays = len(history)
    if plays > 3:
      terminalPass = (history[plays-1] == "p")
      doubleBet = (history[plays-2 : plays] == "bb")

      return terminalPass or doubleBet
    else:
      return False

   #terminal stateかどうかを判定
  def whether_chance_node(self, history):
  
    if history == "":
      return True
    else:
      return False


  # make node or get node
  def if_nonexistant(self, infoSet):
    if infoSet not in self.avg_strategy:
      self.avg_strategy[infoSet] = np.array([1/self.NUM_ACTIONS for _ in range(self.NUM_ACTIONS)], dtype=float)


  def show_plot(self, method):
    pass


  def calc_best_response_value(self, avg_strategy, best_response_strategy, best_response_player, history, prob):
      plays = len(history)
      player = plays % 2
      opponent = 1 - player

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, best_response_player)

      elif self.whether_chance_node(history):
        cards = np.array(["J", "Q", "K"])
        cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards)]
        utility_sum = 0
        for cards_i in cards_candicates:
          nextHistory = cards_i[0] + cards_i[1]
          utility_sum +=  (1/len(cards_candicates))* self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, prob)
        return utility_sum


      infoSet = history[player] + history[2:]
      self.if_nonexistant(infoSet)


      if player == best_response_player:
        if infoSet not in best_response_strategy:
          action_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          br_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)


          for assume_history, po_ in self.infoSets_dict[infoSet]:
            for ai in range(self.NUM_ACTIONS):
              nextHistory =  assume_history + ("p" if ai == 0 else "b")
              br_value[ai] = self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, po_)
              action_value[ai] += br_value[ai] * po_


          br_action = 0
          for ai in range(self.NUM_ACTIONS):
            if action_value[ai] > action_value[br_action]:
              br_action = ai
          best_response_strategy[infoSet] = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          best_response_strategy[infoSet][br_action] = 1.0


        node_util = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          node_util[ai] = self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, prob)
        best_response_util = 0
        for ai in range(self.NUM_ACTIONS):
          best_response_util += node_util[ai] * best_response_strategy[infoSet][ai]

        return best_response_util

      else:
        nodeUtil = 0
        action_value_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          action_value_list[ai] = self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, prob*avg_strategy[infoSet][ai])
          nodeUtil += avg_strategy[infoSet][ai] * action_value_list[ai]
        return nodeUtil


  def create_infoSets(self, history, target_player, po):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    if self.whether_terminal_states(history):
      return

    elif self.whether_chance_node(history):
      cards = np.array(["J", "Q", "K"])
      cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards)]
      for cards_i in cards_candicates:
        nextHistory = cards_i[0] + cards_i[1]
        self.create_infoSets(nextHistory, target_player, po)
      return

    infoSet = history[player] + history[2:]
    if player == target_player:
      if self.infoSets_dict.get(infoSet) is None:
        self.infoSets_dict[infoSet] = []
      self.infoSets_dict[infoSet].append((history, po))

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")
      if player == target_player:
        self.create_infoSets(nextHistory, target_player, po)
      else:
        self.if_nonexistant(infoSet)
        actionProb = self.avg_strategy[infoSet][ai]
        self.create_infoSets(nextHistory, target_player, po*actionProb)


  def update_avg_starategy(self, pi_strategy, be_strategy, iteration_t, lambda_num):
    alpha_t1 = 1 / (iteration_t+2)

    if lambda_num == 1:
        for information_set_u in self.infoSets_dict.keys():
          pi_strategy[information_set_u] +=  alpha_t1*(be_strategy[information_set_u] - pi_strategy[information_set_u])

    elif lambda_num == 2:
      for information_set_u in self.infoSets_dict.keys():
        x_be = self.calculate_realization_plan(information_set_u, be_strategy, 0)
        x_pi = self.calculate_realization_plan(information_set_u, pi_strategy, 1)
        pi_strategy[information_set_u] +=  (alpha_t1*x_be*(be_strategy[information_set_u] - pi_strategy[information_set_u])) / ( (1-alpha_t1)*x_pi + alpha_t1*x_be)



  def calculate_realization_plan(self, information_set_u, target_strategy, bit):
    if len(information_set_u) <= 2:
      return 1
    else:
      hi_a = information_set_u[0]
      if bit == 0:
        return target_strategy[hi_a][0]
      else:
        return target_strategy[hi_a][0]


  def get_exploitability_dfs(self):

    # 各information setを作成 & reach_probabilityを計算
    self.infoSets_dict = {}
    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    exploitability = 0
    best_response_strategy = {}
    for best_response_player_i in range(self.NUM_PLAYERS):
        exploitability += self.calc_best_response_value(self.avg_strategy, best_response_strategy, best_response_player_i, "", 1)

    #assert exploitability >= 0
    return exploitability


  def eval_vanilla_CFR(self, history, target_player_i, iteration_t, p0, p1):
      plays = len(history)
      player = plays % 2
      opponent = 1 - player

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, target_player_i)

      elif self.whether_chance_node(history):
        cards = np.array(["J", "Q", "K"])
        cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards)]
        utility_sum = 0
        for cards_i in cards_candicates:
          nextHistory = cards_i[0] + cards_i[1]
          #regret　strategy 重み どのカードでも同じ確率
          utility_sum += (1/len(cards_candicates))* self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p0, p1)
        return utility_sum

      infoSet = history[player] + history[2:]
      self.if_nonexistant(infoSet)

      strategy = self.avg_strategy[infoSet]

      util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
      nodeUtil = 0

      for ai in range(self.NUM_ACTIONS):
        nextHistory = history + ("p" if ai == 0 else "b")
        if player == 0:
          util_list[ai] = self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p0 * strategy[ai], p1)
        else:
          util_list[ai] = self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p0, p1 * strategy[ai])
        nodeUtil += strategy[ai] * util_list[ai]

      return nodeUtil



  #KuhnTrainer main method
  def train(self, lambda_num):
    self.exploitability_list = {}

    for iteration_t in tqdm(range(int(self.train_iterations))):
      best_response_strategy = {}
      # 各information setを作成 & reach_probabilityを計算
      self.infoSets_dict = {}
      for target_player in range(self.NUM_PLAYERS):
        self.create_infoSets("", target_player, 1.0)

      for best_response_player_i in range(self.NUM_PLAYERS):
        self.calc_best_response_value(self.avg_strategy, best_response_strategy, best_response_player_i, "", 1)


      self.update_avg_starategy(self.avg_strategy, best_response_strategy, iteration_t, lambda_num)


      if iteration_t%20 == 0 and iteration_t != 0:  
    #   if iteration_t in [int(j)-1 for j in np.logspace(1, len(str(self.train_iterations))-1, (len(str(self.train_iterations))-1)*3)] :
        self.exploitability_list[iteration_t] = self.get_exploitability_dfs()

        result_dictionary = {}
        for key, value in sorted(kuhn_trainer_5.avg_strategy.items()):
         result_dictionary[key] = value
        
        difference = 0
        for key in result_dictionary: 
          difference = difference + abs(nash_equilibrium[key][0] - result_dictionary[key][0])
        
        self.differences[iteration_t] = difference

      # if(iteration_t == 10  or iteration_t == 1000 or iteration_t == 5000):
      #   result_dictionary = {}
      #   for key, value in sorted(self.avg_strategy.items()):
      #    result_dictionary[key] = value
        
      #   df = pd.DataFrame(result_dictionary.values(), index=result_dictionary.keys(), columns=['Pass', "Bet"])
      #   df.index.name = "Node"
      #   print("Stratergy after {} iterations is:".format(iteration_t))
      #   print(df)

    self.show_plot("XFP_{}".format(lambda_num))

shortcuts = {"epsilon-greedy" : "eg", "boltzmann": "bz", "dfs": "dfs", "cnt": "cnt", "mlp": "mlp", "general_FSP":"gen", "batch_FSP":"bth", "XFP":"XFP"}

#config
config = dict(
  random_seed = [1, 10, 100, 42][2],
  iterations = 100,
  num_players = 2,
  n= 2,
  m= 1,
  memory_size_rl= 10**3,
  memory_size_sl= 10**3,
  rl_algo = ["epsilon-greedy", "boltzmann", "dfs"][0],
  sl_algo = ["cnt", "mlp"][0],
  pseudo_code = ["general_FSP", "batch_FSP"][1],
  wandb_save = [True, False][1],
  save_matplotlib = [True, False][0],
)


time_taken = []
algorithms_used = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "XFP"]

for i in range(0,3):
  for j in range(0,2):
    for k in range(0,2):

        kuhn_sub_trainer = FSP_Kuhn_Poker_trainer.KuhnTrainer(
        random_seed = config["random_seed"],
        train_iterations = config["iterations"],
        num_players= config["num_players"],
        save_matplotlib = config["save_matplotlib"],
        ) 

        rl_algorithm = ["epsilon-greedy", "boltzmann", "dfs"]
        sl_algorithm = ["cnt", "mlp"]
        pseudo_coder = ["general_FSP", "batch_FSP"]


        start_time = time.time()
        kuhn_sub_trainer.train(
        n = config["n"],
        m = config["m"],
        memory_size_rl = config["memory_size_rl"],
        memory_size_sl = config["memory_size_sl"],
        rl_algo = rl_algorithm[i],
        sl_algo = sl_algorithm[j],
        pseudo_code = pseudo_coder[k],
        wandb_save = config["wandb_save"]
        )

        print("Results for rl algorithm: {}, sl_algorithm: {} and pseudo_code: {} are:".format(rl_algorithm[i], sl_algorithm[j], pseudo_coder[k]))

        result_dict_avg = {}
        for key, value in sorted(kuhn_sub_trainer.avg_strategy.items()):
          result_dict_avg[key] = value
        
        df = pd.DataFrame(result_dict_avg.values(), index=result_dict_avg.keys(), columns=['Pass_avg', "Bet_avg"])
        df.index.name = "Node"

        result_dict_br = {}
        for key, value in sorted(kuhn_sub_trainer.best_response_strategy.items()):
          result_dict_br[key] = value
        df1 = pd.DataFrame(result_dict_br.values(), index=result_dict_br.keys(), columns=['Pass_br', "Bet_br"])
        df1.index.name = "Node"

        df2 = pd.concat([df, df1], axis=1)
        print(df2)

        # print("Time taken is: {}".format((time.time()-start_time)/config["iterations"]))
        time_taken.append((time.time()-start_time)/config["iterations"])

        x = []
        temp = []
        a = []

        # print(kuhn_sub_trainer.exploitability_list)
      

        for key in kuhn_sub_trainer.exploitability_list:
          x.append(key)
          temp.append(kuhn_sub_trainer.exploitability_list[key])
        # for key in kuhn_sub_trainer.differences:
        #   x.append(key)
        #   a.append(kuhn_sub_trainer.differences[key])

        plt.plot(x, temp, label = "FSP : {} : {} : {}".format(rl_algorithm[i], sl_algorithm[j], pseudo_coder[k]))

        # for num in range(2,6):
        #   kuhn_sub_trainer = FSP_Kuhn_Poker_trainer.KuhnTrainer(
        #   random_seed = config["random_seed"],
        #   train_iterations = 80,
        #   num_players= num,
        #   save_matplotlib = config["save_matplotlib"],
        #   ) 

        #   rl_algorithm = ["epsilon-greedy", "boltzmann", "dfs"]
        #   sl_algorithm = ["cnt", "mlp"]
        #   pseudo_coder = ["general_FSP", "batch_FSP"]


        #   start_time = time.time()
        #   kuhn_sub_trainer.train(
        #   n = config["n"],
        #   m = config["m"],
        #   memory_size_rl = config["memory_size_rl"],
        #   memory_size_sl = config["memory_size_sl"],
        #   rl_algo = rl_algorithm[i],
        #   sl_algo = sl_algorithm[j],
        #   pseudo_code = pseudo_coder[k],
        #   wandb_save = config["wandb_save"]
        #   )

        #   print("Results for rl algorithm: {}, sl_algorithm: {} and pseudo_code: {} are:".format(rl_algorithm[i], sl_algorithm[j], pseudo_coder[k]))

        #   result_dict_avg = {}
        #   for key, value in sorted(kuhn_sub_trainer.avg_strategy.items()):
        #     result_dict_avg[key] = value
        #   df = pd.DataFrame(result_dict_avg.values(), index=result_dict_avg.keys(), columns=['Pass_avg', "Bet_avg"])
        #   df.index.name = "Node"

        #   result_dict_br = {}
        #   for key, value in sorted(kuhn_sub_trainer.best_response_strategy.items()):
        #     result_dict_br[key] = value
        #   df1 = pd.DataFrame(result_dict_br.values(), index=result_dict_br.keys(), columns=['Pass_br', "Bet_br"])
        #   df1.index.name = "Node"

        #   df2 = pd.concat([df, df1], axis=1)
        #   print(df2)

        #   # print("Time taken is: {}".format((time.time()-start_time)/config["iterations"]))
        #   # time_taken.append((time.time()-start_time)/config["iterations"])

        #   x = []

        #   # print(kuhn_sub_trainer.exploitability_list)

        #   for key in kuhn_sub_trainer.exploitability_list:
        #     x.append(kuhn_sub_trainer.exploitability_list[key])

        #   print(x)
        #   temp.append(x[-1])
        #   x_final.append(num)


# plt.plot(x_final, temp)
# plt.xlabel("Order of Game States/Population")
# plt.ylabel("Exploitability")
# plt.show()


# x = []
# temp = []

# for key in kuhn_trainer.exploitability_list:
#   x.append(key)
#   temp.append(kuhn_trainer.exploitability_list[key])
# plt.plot(x, temp, label = "FSP")

# start_time = time.time()

kuhn_trainer_5 = KuhnTrainer_XFP(config["iterations"])
kuhn_trainer_5.train(2)

result_dict = {}

for key, value in sorted(kuhn_trainer_5.avg_strategy.items()):
  result_dict[key] = value

df = pd.DataFrame(result_dict.values(), index=result_dict.keys(), columns=['Pass', "Bet"])
df.index.name = "Node"

print(df)
time_taken.append((time.time()-start_time)/config["iterations"])

x = []
y = []
a = []

for key in kuhn_trainer_5.exploitability_list:
  x.append(key)
  y.append(kuhn_trainer_5.exploitability_list[key])

for key in kuhn_trainer_5.differences:
  a.append(kuhn_trainer_5.differences[key])


# # plt.bar(algorithms_used, time_taken, width=0.5)
plt.plot(x, y, label = "XFP")

plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper center')
plt.xlabel("Iterations")
# plt.ylabel("Deviation from nash equilibrium")
plt.ylabel("Exploitability")
plt.show()


doctest.testmod()
