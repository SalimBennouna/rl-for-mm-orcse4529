
import argparse
import numpy as np
import pandas as pd
import datetime as dt

from SimulationCore import SimulationCore
from order import LimitOrder
from Oracle import Oracle

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

cli_parser = argparse.ArgumentParser(description='Simulation configuration.')

cli_parser.add_argument('-c',
                        '--config',
                        required=True,
                        help='Name of config file to execute')
cli_parser.add_argument('--start-time',
                        default='09:30:00',
                        type=str,
                        help='Starting time of simulation (timedelta string, e.g., HH:MM:SS).'
                        )
cli_parser.add_argument('--end-time',
                        default='11:30:00',
                        type=str,
                        help='Ending time of simulation (timedelta string, e.g., up to 168:00:00).'
                        )
cli_parser.add_argument('-l',
                        '--log_dir',
                        default=None,
                        help='Log directory name (default: unix timestamp at program start)')
cli_parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=None,
                        help='numpy.random.seed() for simulation')

cli_parser.add_argument('--mm-type',
                        choices=['none', 'simple', 'rl_baseline', 'rl_tabular', 'rl_tabular_2'],
                        default='none',
                        help='Which market maker class to use (or none).')
cli_parser.add_argument('--mm-wake-up-freq',
                        type=str,
                        default='10S'
                        )
cli_parser.add_argument('--mm-size',
                        type=int,
                        default=10,
                        help='Fixed size for market maker orders')

cli_parser.add_argument('--fund-vol',
                        type=float,
                        default=5e-9,
                        help='Volatility of fundamental time series.'
                        )

cli_args, remaining_args = cli_parser.parse_known_args()

log_dir = cli_args.log_dir
seed = cli_args.seed
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

start_delta = pd.to_timedelta(cli_args.start_time)
end_delta = pd.to_timedelta(cli_args.end_time)

exch_log_flag = True
order_log_flag = True
book_freq = 0

start_wall = dt.datetime.now()
print(f"[config] start_wall={start_wall}")
print(f"[config] seed={seed}\n")

base_day = pd.to_datetime('20010101')
open_ts = base_day + start_delta
close_ts = base_day + end_delta
actor_count, actor_pool, actor_labels = 0, [], []

sym_code = 'AAPL'
cash_start = 10000000
fund_base = 100000
kappa_val = np.log(2) / pd.to_timedelta("30min").value
noise_sigma = fund_base / 1000
lambda_rate = 7e-11

symbol_params = {'r_bar': fund_base,
                 'kappa': kappa_val,
                 'sigma_s': 0,
                 'fund_vol': cli_args.fund_vol,
                 'megashock_lambda_a': 2.7e-18,
                 'megashock_mean': 1e3,
                 'megashock_var': 5e4,
                 'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}
symbols = {sym_code: {'fund_vol': cli_args.fund_vol}}

oracle = Oracle(open_ts, close_ts, sym_code, symbol_params)

history_depth = 25000

actor_pool.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=open_ts,
                             mkt_close=close_ts,
                             symbols=[sym_code],
                             log_orders=exch_log_flag,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=history_depth,
                             book_freq=book_freq,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
actor_labels.extend("ExchangeAgent")
actor_count += 1

noise_ct = 0
noise_mkt_open = base_day + pd.to_timedelta("00:10:00")
noise_mkt_close = base_day + pd.to_timedelta("23:50:00")
actor_pool.extend([NoiseAgent(id=j,
                          name="NoiseAgent {}".format(j),
                          type="NoiseAgent",
                          symbol=sym_code,
                          starting_cash=cash_start,
                          log_orders=order_log_flag,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(actor_count, actor_count + noise_ct)])
actor_count += noise_ct
actor_labels.extend(['NoiseAgent'])

num_value = 5
actor_pool.extend([ValueAgent(id=j,
                          name="Value Agent {}".format(j),
                          type="ValueAgent",
                          symbol=sym_code,
                          starting_cash=cash_start,
                          sigma_n=noise_sigma,
                          r_bar=fund_base,
                          kappa=kappa_val,
                          lambda_a=lambda_rate,
                          log_orders=order_log_flag,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(actor_count, actor_count + num_value)])
actor_count += num_value
actor_labels.extend(['ValueAgent'])

num_mm_agents = 0 if cli_args.mm_type == 'none' else 1

def build_market_maker(idx, actor_id):
    rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))

    if cli_args.mm_type == 'simple':
        mm_wake = '3s'
        return MarketMakerAgent(id=actor_id,
                                name="MARKET_MAKER_AGENT_{}".format(idx),
                                type='MarketMakerAgent',
                                symbol=sym_code,
                                starting_cash=cash_start,
                                min_size=cli_args.mm_size,
                                max_size=cli_args.mm_size,
                                wake_up_freq=mm_wake,
                                inventory_limit=100,
                                log_orders=order_log_flag,
                                random_state=rstate)

    if cli_args.mm_type == 'rl_baseline':
        return RLMarketMakerAgent(id=actor_id,
                                  name="RL_MARKET_MAKER_AGENT_{}".format(idx),
                                  type='RLMarketMakerAgent',
                                  symbol=sym_code,
                                  starting_cash=cash_start,
                                  wake_up_freq=cli_args.mm_wake_up_freq,
                                  base_size=cli_args.mm_size,
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
                                  log_orders=order_log_flag,
                                  random_state=rstate)

    if cli_args.mm_type == 'rl_tabular':
        mm_wake = '3s'
        return RLTabularMarketMakerAgent(id=actor_id,
                                         name="RL_TABULAR_MARKET_MAKER_AGENT_{}".format(idx),
                                         type='RLTabularMarketMakerAgent',
                                         symbol=sym_code,
                                         starting_cash=cash_start,
                                         wake_up_freq=mm_wake,
                                         base_size=cli_args.mm_size,
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
                                          log_orders=order_log_flag,
                                          random_state=rstate)

    if cli_args.mm_type == 'rl_tabular_2':
        mm_wake = '3s'
        return RLTabularMarketMakerAgentV2(id=actor_id,
                                           name="RL_TABULAR_MARKET_MAKER_AGENT_V2_{}".format(idx),
                                           type='RLTabularMarketMakerAgentV2',
                                           symbol=sym_code,
                                           starting_cash=cash_start,
                                           wake_up_freq=mm_wake,
                                           base_size=cli_args.mm_size,
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
                                           log_orders=order_log_flag,
                                           random_state=rstate)

    raise ValueError(f"Unknown mm_type {cli_args.mm_type}")

actor_pool.extend([build_market_maker(idx, j)
               for idx, j in enumerate(range(actor_count, actor_count + num_mm_agents))])
actor_count += num_mm_agents
actor_labels.extend([cli_args.mm_type] * num_mm_agents)

num_zi_agents = 0
actor_pool.extend([ZeroIntelligenceAgent(id=j,
                                     name="ZI_AGENT_{}".format(j),
                                     type="ZeroIntelligenceAgent",
                                     symbol=sym_code,
                                     starting_cash=cash_start,
                                     sigma_n=10000,
                                     sigma_s=symbols[sym_code]['fund_vol'],
                                     kappa=kappa_val,
                                     r_bar=fund_base,
                                     q_max=10,
                                     sigma_pv=5e4,
                                     R_min=0,
                                     R_max=100,
                                     eta=1,
                                     lambda_a=1e-12,
                                     log_orders=order_log_flag,
                                     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                               dtype='uint64')))
               for j in range(actor_count, actor_count + num_zi_agents)])
actor_count += num_zi_agents
actor_labels.extend(['ZeroIntelligenceAgent'])

num_hbl_agents = 0
actor_pool.extend([HeuristicBeliefLearningAgent(id=j,
                                            name="HBL_AGENT_{}".format(j),
                                            type="HeuristicBeliefLearningAgent",
                                            symbol=sym_code,
                                            starting_cash=cash_start,
                                            sigma_n=10000,
                                            sigma_s=symbols[sym_code]['fund_vol'],
                                            kappa=kappa_val,
                                            r_bar=fund_base,
                                            q_max=10,
                                            sigma_pv=5e4,
                                            R_min=0,
                                            R_max=100,
                                            eta=1,
                                            lambda_a=1e-12,
                                            L=2,
                                            log_orders=order_log_flag,
                                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                      dtype='uint64')))
               for j in range(actor_count, actor_count + num_hbl_agents)])
actor_count += num_hbl_agents
actor_labels.extend(['HeuristicBeliefLearningAgent'])

num_momentum_agents = 0

actor_pool.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=sym_code,
                             starting_cash=cash_start,
                             min_size=1,
                             max_size=10,
                             wake_up_freq='20s',
                             log_orders=order_log_flag,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(actor_count, actor_count + num_momentum_agents)])
actor_count += num_momentum_agents
actor_labels.extend("MomentumAgent")

kernel = SimulationCore("MMCompare Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64')))

kernelStartTime = base_day
kernelStopTime = close_ts + pd.to_timedelta('00:01:00')

defaultComputationDelay = 50

kernel.launch(agents=actor_pool,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle,
              log_dir=cli_args.log_dir)

simulation_end_time = dt.datetime.now()
print(f"[config] end_wall={simulation_end_time}")
print(f"[config] duration={simulation_end_time - start_wall}")
