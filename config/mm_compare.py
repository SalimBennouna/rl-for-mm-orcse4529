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
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from agent.market_makers.MarketMakerAgent import MarketMakerAgent
from agent.market_makers.SpreadBasedMarketMakerAgent import SpreadBasedMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent
from model.LatencyModel import LatencyModel

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
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('--end-time',
                    default='11:30:00',
                    type=parse,
                    help='Ending time of simulation.'
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
                    choices=['market', 'adaptive', 'spread'],
                    default='market',
                    help='Which market maker class to use.')
parser.add_argument('--mm-pov',
                    type=float,
                    default=0.025
                    )
parser.add_argument('--mm-window-size',
                    type=util.validate_window_size,
                    default='adaptive'
                    )
parser.add_argument('--mm-min-order-size',
                    type=int,
                    default=1
                    )
parser.add_argument('--mm-max-order-size',
                    type=int,
                    default=10
                    )
parser.add_argument('--mm-num-ticks',
                    type=int,
                    default=10
                    )
parser.add_argument('--mm-wake-up-freq',
                    type=str,
                    default='10S'
                    )
parser.add_argument('--mm-skew-beta',
                    type=float,
                    default=0
                    )
parser.add_argument('--mm-level-spacing',
                    type=float,
                    default=5
                    )
parser.add_argument('--mm-spread-alpha',
                    type=float,
                    default=0.75
                    )
parser.add_argument('--mm-backstop-quantity',
                    type=float,
                    default=50000)

parser.add_argument('--fund-vol',
                    type=float,
                    default=1e-8,
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
mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in CENTS.

r_bar = 1e5
sigma_n = r_bar / 10
kappa = 1.67e-15
lambda_a = 7e-11

# Oracle
symbols = {symbol: {'r_bar': r_bar,
                    'kappa': 1.67e-16,
                    'sigma_s': 0,
                    'fund_vol': args.fund_vol,
                    'megashock_lambda_a': 2.77778e-18,
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
num_noise = 5000
noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")  # These times needed for distribution of arrival times
                                                                # of Noise Agents
noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
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
num_value = 100
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

"""
window_size ==  Spread of market maker (in ticks) around the mid price
pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
       the market maker places at each level
num_ticks == Number of levels to place orders in around the spread
wake_up_freq == How often the market maker wakes up

"""

mm_window_size = args.mm_window_size
if args.mm_type == 'spread':
    try:
        mm_window_size = int(mm_window_size)
    except ValueError as exc:
        raise ValueError("--mm-window-size must be an integer when mm-type is spread.") from exc

# Single market maker tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
mm_params = [(mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size)]

num_mm_agents = 1
mm_cancel_limit_delay = 50  # 50 nanoseconds

def build_market_maker(idx, agent_id):
    rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))
    window_size, pov, num_ticks, wake_up_freq, min_order_size = mm_params[idx]

    if args.mm_type == 'market':
        return MarketMakerAgent(id=agent_id,
                                name="MARKET_MAKER_AGENT_{}".format(idx),
                                type='MarketMakerAgent',
                                symbol=symbol,
                                starting_cash=starting_cash,
                                min_size=args.mm_min_order_size,
                                max_size=args.mm_max_order_size,
                                wake_up_freq=wake_up_freq,
                                subscribe=False,
                                log_orders=log_orders,
                                random_state=rstate)

    if args.mm_type == 'adaptive':
        return AdaptiveMarketMakerAgent(id=agent_id,
                                        name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(idx),
                                        type='AdaptivePOVMarketMakerAgent',
                                        symbol=symbol,
                                        starting_cash=starting_cash,
                                        pov=pov,
                                        min_order_size=min_order_size,
                                        window_size=window_size,
                                        num_ticks=num_ticks,
                                        wake_up_freq=wake_up_freq,
                                        cancel_limit_delay=mm_cancel_limit_delay,
                                        skew_beta=args.mm_skew_beta,
                                        level_spacing=args.mm_level_spacing,
                                        spread_alpha=args.mm_spread_alpha,
                                        backstop_quantity=args.mm_backstop_quantity,
                                        log_orders=log_orders,
                                        random_state=rstate)

    # args.mm_type == 'spread'
    if isinstance(window_size, str):
        raise ValueError("--mm-window-size must be numeric when using spread-based market makers.")
    return SpreadBasedMarketMakerAgent(id=agent_id,
                                       name="SPREAD_MARKET_MAKER_AGENT_{}".format(idx),
                                       type='SpreadBasedMarketMakerAgent',
                                       symbol=symbol,
                                       starting_cash=starting_cash,
                                       order_size=min_order_size,
                                       window_size=window_size,
                                       num_ticks=num_ticks,
                                       wake_up_freq=wake_up_freq,
                                       subscribe=False,
                                       log_orders=log_orders,
                                       random_state=rstate)


agents.extend([build_market_maker(idx, j)
               for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
agent_count += num_mm_agents
agent_types.extend([args.mm_type] * num_mm_agents)


# 5) Zero Intelligence Agents
num_zi_agents = 50
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
num_hbl_agents = 25
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
num_momentum_agents = 25

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

# LATENCY

latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**32))
pairwise = (agent_count, agent_count)

# All agents sit on line from Seattle to NYC
nyc_to_seattle_meters = 3866660
pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                        random_state=latency_rstate)
pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

model_args = {
    'connected': True,
    'min_latency': pairwise_latencies
}

latency_model = LatencyModel(latency_model='deterministic',
                             random_state=latency_rstate,
                             kwargs=model_args
                             )
# KERNEL

kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              agentLatencyModel=latency_model,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle,
              log_dir=args.log_dir)


simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
