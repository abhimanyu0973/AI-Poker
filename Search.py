#Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
import doctest
import copy
import math
import wandb
import time
from tqdm import tqdm
from collections import defaultdict


nash_equilibrium = {'J': np.array([0.8, 0.2]), 'Jb': np.array([1,0]) , 'Jp': np.array([0.67, 0.33]), 'Jpb': np.array([1, 0]), 
                    'K': np.array([0.25, 0.75]), 'Kb': np.array([0,1]), 'Kp': np.array([0,1]), 'Kpb': np.array([0,1]),
                      'Q': np.array([1,0]), 'Qb': np.array([0.67, 0.33]), 'Qp': np.array([1,0]), 'Qpb': np.array([0.4, 0.6])}

#Node Class
class Node:
  #Kuhn_node_definitions
  def __init__(self, NUM_ACTIONS):
    self.NUM_ACTIONS = NUM_ACTIONS
    self.infoSet = None
    self.c = 0
    self.regretSum = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.strategy = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.strategySum = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.Get_strategy_through_regret_matching()


  #regret-matching
  def Get_strategy_through_regret_matching(self):
    self.normalizingSum = 0
    for a in range(self.NUM_ACTIONS):
      self.strategy[a] = self.regretSum[a] if self.regretSum[a]>0 else 0
      self.normalizingSum += self.strategy[a]

    for a in range(self.NUM_ACTIONS):
      if self.normalizingSum >0 :
        self.strategy[a] /= self.normalizingSum
      else:
        self.strategy[a] = 1/self.NUM_ACTIONS


  # calculate average-strategy
  def Get_average_information_set_mixed_strategy(self):
    self.avgStrategy = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.normalizingSum = 0
    for a in range(self.NUM_ACTIONS):
      self.normalizingSum += self.strategySum[a]

    for a in range(self.NUM_ACTIONS):
      if self.normalizingSum >0 :
        self.avgStrategy[a] = self.strategySum[a] / self.normalizingSum
      else:
        self.avgStrategy[a] = 1/ self.NUM_ACTIONS

    return self.avgStrategy



#Trainer class
class KuhnTrainer:
  def __init__(self, train_iterations=10**4, num_players =2, random_seed = 42, save_matplotlib = False):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.NUM_ACTIONS = 2
    self.nodeMap = defaultdict(list)
    self.eval = False
    self.card_rank = self.make_rank(self.NUM_PLAYERS)
    self.random_seed = random_seed
    self.differences = {}

    self.random_seed_fix(self.random_seed)

    self.save_matplotlib = save_matplotlib

    self.exploitability_time = 0

  def random_seed_fix(self, random_seed):
      random.seed(random_seed)
      np.random.seed(random_seed)


  def make_rank(self, num_players):

    card_rank = {}
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    for i in range(num_players+1):
      card_rank[card[11-num_players+i]] =  i+1
    return card_rank


  def card_distribution(self, num_players):
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    return card[11-num_players:]


  #return util for terminal state for target_player_i
  def Return_payoff_for_terminal_states(self, history, target_player_i):
      
      pot = self.NUM_PLAYERS * 1 + history.count("b")
      start = -1
      target_player_action = history[self.NUM_PLAYERS+target_player_i::self.NUM_PLAYERS]

      # all players pass
      if ("b" not in history) and (history.count("p") == self.NUM_PLAYERS):
        pass_player_card = {}
        for idx in range(self.NUM_PLAYERS):
          pass_player_card[idx] = [history[idx], self.card_rank[history[idx]]]

        winner_rank = max([idx[1] for idx in pass_player_card.values()])

        target_player_rank = pass_player_card[target_player_i][1]

        if target_player_rank == winner_rank:
          return start + pot
        else:
          return start

      #target plyaer do pass , another player do bet
      elif ("b" not in target_player_action) and ("b" in history):
        return start

      else:
        #bet → +pot or -2
        bet_player_list = [idx%self.NUM_PLAYERS for idx, act in enumerate(history[self.NUM_PLAYERS:]) if act == "b"]
        bet_player_card = {}
        for idx in bet_player_list:
          bet_player_card[idx] = [history[idx], self.card_rank[history[idx]]]

        winner_rank = max([idx[1] for idx in bet_player_card.values()])
        target_player_rank = bet_player_card[target_player_i][1]
        if target_player_rank == winner_rank:
          return start + pot - 1
        else:
          return start - 1


  #whether terminal state
  def whether_terminal_states(self, history):
    #pass only history
    if "b" not in history:
      return history.count("p") == self.NUM_PLAYERS

    plays = len(history)
    first_bet = history.index("b")
    return plays - first_bet -1  == self.NUM_PLAYERS -1


  #whether chance node state
  def whether_chance_node(self, history):
    if history == "":
      return True
    else:
      return False


  # make node or get node
  def Get_information_set_node_or_create_it_if_nonexistant(self, infoSet):
    node = self.nodeMap.get(infoSet)
    if node == None:
      node = Node(self.NUM_ACTIONS)
      self.nodeMap[infoSet] = node
    return node


  #chance sampling CFR
  def chance_sampling_CFR(self, history, target_player_i, iteration_t, p_list):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      random.shuffle(cards)
      nextHistory = "".join(cards[:self.NUM_PLAYERS])
      return self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p_list)

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0
    node.Get_strategy_through_regret_matching()

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")

      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = node.strategy[ai]

      util_list[ai] = self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p_list * p_change)
      nodeUtil += node.strategy[ai] * util_list[ai]


    if player == target_player_i:
      for ai in range(self.NUM_ACTIONS):
        regret = util_list[ai] - nodeUtil

        p_exclude = 1
        for idx in range(self.NUM_PLAYERS):
          if idx != player:
            p_exclude *= p_list[idx]

        node.regretSum[ai] += p_exclude * regret
        node.strategySum[ai] += node.strategy[ai] * p_list[player]

    return nodeUtil


  def vanilla_CFR(self, history, target_player_i, iteration_t, p_list):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
      utility_sum = 0
      for cards_i in cards_candicates:
        nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
        utility_sum +=  (1/len(cards_candicates))* self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list)
      return utility_sum

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    node.Get_strategy_through_regret_matching()

    if not self.eval:
      strategy =  node.strategy
    else:
      strategy = node.Get_average_information_set_mixed_strategy()


    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")

      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = strategy[ai]

      util_list[ai] = self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list * p_change)

      nodeUtil += strategy[ai] * util_list[ai]

    if (not self.eval) and  player == target_player_i:
      for ai in range(self.NUM_ACTIONS):
        regret = util_list[ai] - nodeUtil

        p_exclude = 1
        for idx in range(self.NUM_PLAYERS):
          if idx != player:
            p_exclude *= p_list[idx]

        node.regretSum[ai] += p_exclude * regret
        node.strategySum[ai] += strategy[ai] * p_list[player]

    return nodeUtil


  #external sampling MCCFR
  def external_sampling_MCCFR(self, history, target_player_i):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      random.shuffle(cards)
      nextHistory = "".join(cards[:self.NUM_PLAYERS])
      return self.external_sampling_MCCFR(nextHistory, target_player_i)

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    node.Get_strategy_through_regret_matching()

    if player == target_player_i:
      util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
      nodeUtil = 0

      for ai in range(self.NUM_ACTIONS):
        nextHistory = history + ("p" if ai == 0 else "b")
        util_list[ai] = self.external_sampling_MCCFR(nextHistory, target_player_i)
        nodeUtil += node.strategy[ai] * util_list[ai]

      for ai in range(self.NUM_ACTIONS):
        regret = util_list[ai] - nodeUtil
        node.regretSum[ai] += regret

    else:
      sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p= node.strategy)
      nextHistory = history + ("p" if sampling_action == 0 else "b")
      nodeUtil= self.external_sampling_MCCFR(nextHistory, target_player_i)

      for ai in range(self.NUM_ACTIONS):
        node.strategySum[ai] += node.strategy[ai]

    return nodeUtil


  #outcome sampling MCCFR
  def outcome_sampling_MCCFR(self, history, target_player_i, iteration_t, p_list,s):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i) / s, 1

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      random.shuffle(cards)
      nextHistory = "".join(cards[:self.NUM_PLAYERS])
      return self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p_list, s)

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    node.Get_strategy_through_regret_matching()
    probability =  np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)

    if player == target_player_i:
      for ai in range(self.NUM_ACTIONS):
        probability[ai] =  self.epsilon/self.NUM_ACTIONS + (1-self.epsilon)* node.strategy[ai]
    else:
      for ai in range(self.NUM_ACTIONS):
        probability[ai] = node.strategy[ai]

    sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p=probability)
    nextHistory = history + ("p" if sampling_action == 0 else "b")

    if player == target_player_i:

      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = node.strategy[sampling_action]

      util, p_tail = self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p_list*p_change, s*probability[sampling_action])

      p_exclude = 1
      for idx in range(self.NUM_PLAYERS):
        if idx != player:
          p_exclude *= p_list[idx]

      w = util * p_exclude
      for ai in range(self.NUM_ACTIONS):
        if sampling_action == ai:
          regret = w*(1- node.strategy[sampling_action])*p_tail
        else:
          regret = -w*p_tail * node.strategy[sampling_action]
        node.regretSum[ai] +=  regret

    else:
        p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
        for idx in range(self.NUM_PLAYERS):
          if idx!= player:
            p_change[idx] = node.strategy[sampling_action]

        util, p_tail = self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p_list*p_change, s*probability[sampling_action])

        p_exclude = 1
        for idx in range(self.NUM_PLAYERS):
          if idx != player:
            p_exclude *= p_list[idx]

        for ai in range(self.NUM_ACTIONS):
          node.strategySum[ai] += (iteration_t - node.c)*p_exclude*node.strategy[ai]
        node.c = iteration_t
        #node.strategySum[ai] += (p1/s)*node.strategy[ai]

    return util, p_tail*node.strategy[sampling_action]


  #KuhnTrainer main method
  def train(self, method):
    self.exploitability_list = {}
    self.avg_utility_list = {}

    #matplotlib
    if self.save_matplotlib:
      self.ex_name = "exploitability_for_{}_{}".format(self.random_seed, method)
      self.database_for_plot = {"iteration":[] ,self.ex_name:[]}


    for iteration_t in tqdm(range(1, int(self.train_iterations)+1)):
      for target_player_i in range(self.NUM_PLAYERS):

        p_list = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)

        if method == "vanilla_CFR":
          self.vanilla_CFR("", target_player_i, iteration_t, p_list)
        elif method == "chance_sampling_CFR":
          self.chance_sampling_CFR("", target_player_i, iteration_t, p_list)
        elif method == "external_sampling_MCCFR":
          self.external_sampling_MCCFR("", target_player_i)
        elif method == "outcome_sampling_MCCFR":
          self.epsilon = 0.6
          self.outcome_sampling_MCCFR("", target_player_i, iteration_t, p_list, 1)

      start_calc_exploitability = time.time()
      #calculate expolitability
      # if iteration_t in [int(j) for j in np.logspace(0, len(str(self.train_iterations)), (len(str(self.train_iterations)))*10 , endpoint=False)] :
      if iteration_t%10 == 0:
        self.exploitability_list[iteration_t] = self.get_exploitability_dfs()
        self.avg_utility_list[iteration_t] = self.eval_strategy(target_player_i=0)
#----------------------------------------------------------------------------------------------
        result_dictionary = {}
        difference = 0
        for key, value in sorted(self.nodeMap.items()):
          result_dictionary[key] = value.Get_average_information_set_mixed_strategy()
        for key in result_dictionary:
          difference = difference + abs(nash_equilibrium[key][0] - result_dictionary[key][0])
          self.differences[iteration_t] = difference
        

        #追加 matplotlibで図を書くため
        if self.save_matplotlib:
          self.database_for_plot["iteration"].append(iteration_t)
          self.database_for_plot[self.ex_name].append(self.exploitability_list[iteration_t])
      
      # if(iteration_t == 10 or iteration_t == 1000 or iteration_t == 5000 or iteration_t == 10000):
      #   result_dictionary = {}
      #   for key, value in sorted(self.nodeMap.items()):
      #    result_dictionary[key] = value.Get_average_information_set_mixed_strategy()
        
      #   print(result_dictionary)
      #   df = pd.DataFrame(result_dictionary.values(), index=result_dictionary.keys(), columns=['Pass', "Bet"])
      #   df.index.name = "Node"
      #   print("Stratergy after {} iterations is:".format(iteration_t))
      #   print(df)

      end_calc_exploitability = time.time()
      self.exploitability_time += end_calc_exploitability - start_calc_exploitability



  # evaluate average strategy
  def eval_strategy(self, target_player_i):
    self.eval = True
    p_list = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
    average_utility = self.vanilla_CFR("", target_player_i, 0, p_list)
    self.eval = False
    return average_utility


  def calc_best_response_value(self, best_response_strategy, best_response_player, history, prob):
      plays = len(history)
      player = plays % self.NUM_PLAYERS

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, best_response_player)

      elif self.whether_chance_node(history):
        cards = self.card_distribution(self.NUM_PLAYERS)
        cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
        utility_sum = 0
        for cards_i in cards_candicates:
          nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
          utility_sum +=  (1/len(cards_candicates))* self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)
        return utility_sum

      infoSet = history[player] + history[self.NUM_PLAYERS:]
      node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)

      if player == best_response_player:
        if infoSet not in best_response_strategy:
          action_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          br_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)


          for assume_history, po_ in self.infoSets_dict[infoSet]:
            for ai in range(self.NUM_ACTIONS):
              nextHistory =  assume_history + ("p" if ai == 0 else "b")
              br_value[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, po_)
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
          node_util[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)
        best_response_util = 0
        for ai in range(self.NUM_ACTIONS):
          best_response_util += node_util[ai] * best_response_strategy[infoSet][ai]

        return best_response_util

      else:
        avg_strategy = node.Get_average_information_set_mixed_strategy()
        nodeUtil = 0
        action_value_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          action_value_list[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob*avg_strategy[ai])
          nodeUtil += avg_strategy[ai] * action_value_list[ai]
        return nodeUtil


  def create_infoSets(self, history, target_player, po):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
      for cards_i in cards_candicates:
        nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
        self.create_infoSets(nextHistory, target_player, po)
      return

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    if player == target_player:
      if self.infoSets_dict.get(infoSet) is None:
        self.infoSets_dict[infoSet] = []
      self.infoSets_dict[infoSet].append((history, po))

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")
      if player == target_player:
        self.create_infoSets(nextHistory, target_player, po)
      else:
        node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
        actionProb = node.Get_average_information_set_mixed_strategy()[ai]
        self.create_infoSets(nextHistory, target_player, po*actionProb)


  def get_exploitability_dfs(self):
    # make each information set & calculate reach_probability
    self.infoSets_dict = {}
    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    exploitability = 0
    best_response_strategy = {}
    for best_response_player_i in range(self.NUM_PLAYERS):
        exploitability += self.calc_best_response_value(best_response_strategy, best_response_player_i, "", 1)

    assert exploitability >= 0
    return exploitability

#----------------------------------------------------------------------------------------------------------------------------------------------------
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


      if iteration_t%10 == 0:  
    #   if iteration_t in [int(j)-1 for j in np.logspace(1, len(str(self.train_iterations))-1, (len(str(self.train_iterations))-1)*3)] :
        self.exploitability_list[iteration_t] = self.get_exploitability_dfs()

        result_dictionary = {}
        for key, value in sorted(kuhn_trainer_5.avg_strategy.items()):
         result_dictionary[key] = value
        
        difference = 0
        for key in result_dictionary: 
          difference = difference + abs(nash_equilibrium[key][0] - result_dictionary[key][0])
        
        self.differences[iteration_t] = difference

      if(iteration_t == 10  or iteration_t == 1000 or iteration_t == 5000):
        result_dictionary = {}
        for key, value in sorted(self.avg_strategy.items()):
         result_dictionary[key] = value
        
        df = pd.DataFrame(result_dictionary.values(), index=result_dictionary.keys(), columns=['Pass', "Bet"])
        df.index.name = "Node"
        print("Stratergy after {} iterations is:".format(iteration_t))
        print(df)

    self.show_plot("XFP_{}".format(lambda_num))

config = dict(
  algo = ["vanilla_CFR", "chance_sampling_CFR", "external_sampling_MCCFR", "outcome_sampling_MCCFR"][3],
  train_iterations = 1000,
  num_players =  2,
  random_seed = [1, 10, 100, 42][3],
  wandb_save = False,
  save_matplotlib = [True, False][0],
)

time_taken = []
algorithms_used = ["vanilla_CFR", "chance_sampling_CFR", "external_sampling_MCCFR", "outcome_sampling_MCCFR", "XFP"]

for i in range(0, 4):
  temp = []
  x = []
  a = []
  # difference = 0

  start_time = time.time()

  algorithm = ["vanilla_CFR", "chance_sampling_CFR", "external_sampling_MCCFR", "outcome_sampling_MCCFR"][i]
  kuhn_sub_trainer = KuhnTrainer(
    train_iterations=config["train_iterations"],
    num_players=config["num_players"],
    random_seed=config["random_seed"],
    save_matplotlib = config["save_matplotlib"])
  kuhn_sub_trainer.train(algorithm)

  result_dict = {}
  for key, value in sorted(kuhn_sub_trainer.nodeMap.items()):
    result_dict[key] = value.Get_average_information_set_mixed_strategy()


  df = pd.DataFrame(result_dict.values(), index=result_dict.keys(), columns=['Pass', "Bet"])
  df.index.name = "Node"
  print(df)
  time_taken.append((time.time()-start_time)/config["train_iterations"])

  for key in kuhn_sub_trainer.differences:
    a.append(kuhn_sub_trainer.differences[key])
  
    
  for key in kuhn_sub_trainer.database_for_plot:
    if(key != 'iteration'):
      temp = kuhn_sub_trainer.database_for_plot[key]
    else:
      x  = kuhn_sub_trainer.database_for_plot[key]

  if(i >= 0):
    plt.plot(x, a, label = algorithm)

start_time = time.time()

kuhn_trainer_5 = KuhnTrainer_XFP(config["train_iterations"])
kuhn_trainer_5.train(2)

result_dict = {}

for key, value in sorted(kuhn_trainer_5.avg_strategy.items()):
  result_dict[key] = value

df = pd.DataFrame(result_dict.values(), index=result_dict.keys(), columns=['Pass', "Bet"])
df.index.name = "Node"

print(df)
time_taken.append((time.time()-start_time)/10000)

x = []
y = []
a = []

for key in kuhn_trainer_5.exploitability_list:
  x.append(key)
  y.append(kuhn_trainer_5.exploitability_list[key])

for key in kuhn_trainer_5.differences:
  a.append(kuhn_trainer_5.differences[key])

plt.plot(x, a, label = "XFP")
# plt.xscale('log')
# plt.yscale('log')
plt.legend(loc='upper center')
plt.xlabel("Iterations")
# plt.ylabel("Equilibrium Accuracy")
plt.ylabel("Deviation from Equilibrium")
plt.show()

# plt.ylabel("Average Time Taken per Iteration")
# plt.xlabel("Algorithms Used")
# plt.bar(algorithms_used, time_taken)
# plt.show()

# temp = []
# x_final = []

# for i in range(2,6):

#   x = []

#   kuhn_trainer_5 = KuhnTrainer_XFP(config["train_iterations"])
#   kuhn_trainer_5.train(2)

#   kuhn_trainer_5.NUM_PLAYERS = i

#   result_dict = {}

#   for key, value in sorted(kuhn_trainer_5.avg_strategy.items()):
#     result_dict[key] = value

#   df = pd.DataFrame(result_dict.values(), index=result_dict.keys(), columns=['Pass', "Bet"])
#   df.index.name = "Node"
#   print(df)

#   x = []
#   y = []
#   a = []

#   for key in kuhn_trainer_5.exploitability_list:
#     x.append(key)
#     y.append(kuhn_trainer_5.exploitability_list[key])

#   print(x)
#   temp.append(x[-1])
#   x_final.append(i)
  

# print(x_final)
# print(temp)


# plt.xlabel("Order of Game States/Population")
# plt.ylabel("Exploitability")
# plt.plot(x_final, temp, label="vanilla_CFR")
# plt.show()
  

doctest.testmod()