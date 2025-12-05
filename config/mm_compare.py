# RMSC-3 variant (market maker comparison):
# - 1     Exchange Agent
# - 1     Market Maker Agent (type configurable)
# - 100   Value Agents
# - 50    Zero Intelligence Agents
# - 25    Heuristic Belief Learning Agents
# - 25    Momentum Agents
# - 5000  Noise Agents

import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent
from agent.market_makers.MarketMakerAgent import MarketMakerAgent
from agent.market_makers.RLMarketMakerAgent import RLMarketMakerAgent
from agent.market_makers.RLTabularMarketMakerAgent import RLTabularMarketMakerAgent
from agent.market_makers.RLTabularMarketMakerAgentV2 import RLTabularMarketMakerAgentV2
from agent.MomentumAgent import MomentumAgent

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Market maker comparison config (RMSC03 baseline).')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-d', '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('--start-time',
                    default='09:30:00',
                    type=str,
                    help='Starting time of simulation (timedelta string, e.g., HH:MM:SS).'
                    )
parser.add_argument('--end-time',
                    default='11:30:00',
                    type=str,
                    help='Ending time of simulation (timedelta string, e.g., up to 168:00:00).'
                    )
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# Execution agent config
# market maker config
parser.add_argument('--mm-type',
                    choices=['none', 'simple', 'rl_baseline', 'rl_tabular', 'rl_tabular_2'],
                    default='none',
                    help='Which market maker class to use (or none).')
parser.add_argument('--mm-pov',
                    type=float,
                    default=0.025
                    )
parser.add_argument('--mm-min-order-size',
                    type=int,
                    default=1
                    )
parser.add_argument('--mm-max-order-size',
                    type=int,
                    default=10
                    )
parser.add_argument('--mm-wake-up-freq',
                    type=str,
                    default='10S'
                    )
parser.add_argument('--mm-size',
                    type=int,
                    default=10,
                    help='Fixed size for market maker orders')

parser.add_argument('--fund-vol',
                    type=float,
                    default=5e-9,
                    help='Volatility of fundamental time series.'
                    )

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

# Convert start/end times to timedeltas to allow multi-day (e.g., >24h) horizons.
start_delta = pd.to_timedelta(args.start_time)
end_delta = pd.to_timedelta(args.end_time)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

exchange_log_orders = True
log_orders = True
book_freq = 0

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + start_delta
mkt_close = historical_date + end_delta
agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in CENTS.

r_bar = 100000
# Half-life of 30 minutes expressed in nanoseconds to match OU time units.
kappa = np.log(2) / pd.to_timedelta("30min").value
sigma_n = r_bar / 1000  # further lower observation noise for value agents
lambda_a = 7e-11  # slower arrival rate for value agent trades (around ~14s mean)

# Oracle
symbols = {symbol: {'r_bar': r_bar,
                    'kappa': kappa,
                    'sigma_s': 0,
                    'fund_vol': args.fund_vol,
                    'megashock_lambda_a': 2.7e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

# 1) Exchange Agent

#  How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
stream_history_length = 25000

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=exchange_log_orders,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=stream_history_length,
                             book_freq=book_freq,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) Noise Agents
num_noise = 0
noise_mkt_open = historical_date + pd.to_timedelta("00:10:00")  # These times needed for distribution of arrival times
                                                                # of Noise Agents
noise_mkt_close = historical_date + pd.to_timedelta("23:50:00")
agents.extend([NoiseAgent(id=j,
                          name="NoiseAgent {}".format(j),
                          type="NoiseAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_noise)])
agent_count += num_noise
agent_types.extend(['NoiseAgent'])

# 3) Value Agents
num_value = 5
agents.extend([ValueAgent(id=j,
                          name="Value Agent {}".format(j),
                          type="ValueAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          sigma_n=sigma_n,
                          r_bar=r_bar,
                          kappa=kappa,
                          lambda_a=lambda_a,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_value)])
agent_count += num_value
agent_types.extend(['ValueAgent'])

# 4) Market Maker Agents

num_mm_agents = 0 if args.mm_type == 'none' else 1

def build_market_maker(idx, agent_id):
    rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))

    if args.mm_type == 'simple':
        mm_wake = '3s'
        return MarketMakerAgent(id=agent_id,
                                name="MARKET_MAKER_AGENT_{}".format(idx),
                                type='MarketMakerAgent',
                                symbol=symbol,
                                starting_cash=starting_cash,
                                min_size=args.mm_size,
                                max_size=args.mm_size,
                                wake_up_freq=mm_wake,
                                subscribe=False,
                                inventory_limit=100,
                                log_orders=log_orders,
                                random_state=rstate)

    if args.mm_type == 'rl_baseline':
        return RLMarketMakerAgent(id=agent_id,
                                  name="RL_MARKET_MAKER_AGENT_{}".format(idx),
                                  type='RLMarketMakerAgent',
                                  symbol=symbol,
                                  starting_cash=starting_cash,
                                  wake_up_freq=args.mm_wake_up_freq,
                                  base_size=args.mm_size,
                                  offsets=[1, 2, 3],
                                  epsilon=1.0,
                                  alpha=0.1,
                                  gamma=0.95,
                                  inventory_clip=100,
                                  spread_clip=7,
                                  inventory_bin=10,
                                  spread_bin=1,
                                  inventory_penalty=1.0,
                                  inventory_limit=100,
                                  log_orders=log_orders,
                                  random_state=rstate)

    if args.mm_type == 'rl_tabular':
        mm_wake = '3s'
        return RLTabularMarketMakerAgent(id=agent_id,
                                         name="RL_TABULAR_MARKET_MAKER_AGENT_{}".format(idx),
                                         type='RLTabularMarketMakerAgent',
                                         symbol=symbol,
                                         starting_cash=starting_cash,
                                         wake_up_freq=mm_wake,
                                         base_size=args.mm_size,
                                         base_offsets=[5, 15, 25, 35],
                                         epsilon=1.0,
                                         alpha=0.1,
                                          gamma=0.95,
                                          inventory_clip=100,
                                          spread_clip=80,
                                           inventory_bin=10,
                                           spread_bin=5,
                                          inventory_penalty=10.0,
                                          inventory_limit=100,
                                          log_orders=log_orders,
                                          random_state=rstate)

    if args.mm_type == 'rl_tabular_2':
        mm_wake = '3s'
        return RLTabularMarketMakerAgentV2(id=agent_id,
                                           name="RL_TABULAR_MARKET_MAKER_AGENT_V2_{}".format(idx),
                                           type='RLTabularMarketMakerAgentV2',
                                           symbol=symbol,
                                           starting_cash=starting_cash,
                                           wake_up_freq=mm_wake,
                                           base_size=args.mm_size,
                                           base_offsets=[5, 15, 25, 35],
                                           epsilon=1.0,
                                           alpha=0.1,
                                           gamma=0.95,
                                           inventory_clip=100,
                                           spread_clip=80,
                                            inventory_bin=10,
                                            spread_bin=5,
                                           inventory_penalty=10.0,
                                           inventory_limit=100,
                                           log_orders=log_orders,
                                           random_state=rstate)

    raise ValueError(f"Unknown mm_type {args.mm_type}")


agents.extend([build_market_maker(idx, j)
               for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
agent_count += num_mm_agents
agent_types.extend([args.mm_type] * num_mm_agents)


# 5) Zero Intelligence Agents
num_zi_agents = 0
agents.extend([ZeroIntelligenceAgent(id=j,
                                     name="ZI_AGENT_{}".format(j),
                                     type="ZeroIntelligenceAgent",
                                     symbol=symbol,
                                     starting_cash=starting_cash,
                                     sigma_n=10000,
                                     sigma_s=symbols[symbol]['fund_vol'],
                                     kappa=kappa,
                                     r_bar=r_bar,
                                     q_max=10,
                                     sigma_pv=5e4,
                                     R_min=0,
                                     R_max=100,
                                     eta=1,
                                     lambda_a=1e-12,
                                     log_orders=log_orders,
                                     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                               dtype='uint64')))
               for j in range(agent_count, agent_count + num_zi_agents)])
agent_count += num_zi_agents
agent_types.extend(['ZeroIntelligenceAgent'])

# 6) Heuristic Belief Learning Agents
num_hbl_agents = 0
agents.extend([HeuristicBeliefLearningAgent(id=j,
                                            name="HBL_AGENT_{}".format(j),
                                            type="HeuristicBeliefLearningAgent",
                                            symbol=symbol,
                                            starting_cash=starting_cash,
                                            sigma_n=10000,
                                            sigma_s=symbols[symbol]['fund_vol'],
                                            kappa=kappa,
                                            r_bar=r_bar,
                                            q_max=10,
                                            sigma_pv=5e4,
                                            R_min=0,
                                            R_max=100,
                                            eta=1,
                                            lambda_a=1e-12,
                                            L=2,
                                            log_orders=log_orders,
                                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                      dtype='uint64')))
               for j in range(agent_count, agent_count + num_hbl_agents)])
agent_count += num_hbl_agents
agent_types.extend(['HeuristicBeliefLearningAgent'])

# 7) Momentum Agents
num_momentum_agents = 0

agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             wake_up_freq='20s',
                             log_orders=log_orders,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")


########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("MMCompare Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64')))

kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

defaultComputationDelay = 50  # 50 nanoseconds
# KERNEL

kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle,
              log_dir=args.log_dir)


simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
