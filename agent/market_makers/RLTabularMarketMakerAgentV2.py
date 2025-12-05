from agent.TradingAgent import TradingAgent
import pandas as pd
import numpy as np
from util.util import log_print


class RLTabularMarketMakerAgentV2(TradingAgent):
    """
    Tabular SARSA/Q-learning market maker (PnL reward) with inventory+delta_mid state.

    State: (inventory_bin, delta_mid_bin) where delta_mid is last mid change.
    Action: independently choose bid/ask offsets (or skip a side) at fixed distances from mid.

    Reward: delta mark-to-market PnL (no inventory penalty).
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 wake_up_freq='3s',
                 base_size=60,
                 base_offsets=(5, 15, 25, 35),
                 epsilon=0.1,
                 alpha=0.1,
                 gamma=0.95,
                 inventory_clip=100,
                 spread_clip=80,
                 inventory_bin=10,
                 spread_bin=5,
                 delta_mid_clip=50,
                 delta_mid_bin=5,
                 inventory_penalty=10.0,
                 inventory_limit=100,
                 epsilon_half_life_hours=5,
                 log_orders=False,
                 random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.wake_up_freq = wake_up_freq
        self.base_size = base_size
        self.base_offsets = list(base_offsets)
        self.actions = self._build_actions()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.inventory_clip = inventory_clip
        self.spread_clip = spread_clip
        self.inventory_bin = inventory_bin
        self.spread_bin = spread_bin
        self.delta_mid_clip = delta_mid_clip
        self.delta_mid_bin = delta_mid_bin
        self.inventory_penalty = inventory_penalty
        self.inventory_limit = inventory_limit
        self.step_count = 0
        try:
            wake_seconds = pd.Timedelta(self.wake_up_freq).total_seconds()
        except Exception:
            wake_seconds = 1.0
        half_life_seconds = epsilon_half_life_hours * 3600.0
        self.epsilon_decay_steps = max(1, int(np.ceil(half_life_seconds / wake_seconds)))

        self.q = {}
        self.last_state = None
        self.last_action = None
        self.last_mtm = 0
        self.last_mid = None
        self.last_delta_mid = 0
        self.cum_reward = 0
        self.state = 'AWAITING_WAKEUP'

    def _build_actions(self):
        offsets = [None] + self.base_offsets
        return [(bid, ask) for bid in offsets for ask in offsets]

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.last_mtm = self._mark_to_market(None)

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if can_trade:
            self.getCurrentSpread(self.symbol, depth=1)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
            if not (bid and ask):
                log_print(f"{self.name}: missing spread at {currentTime}, skipping quote")
                self.setWakeup(currentTime + self.getWakeFrequency())
                return

            mid = (bid + ask) / 2
            delta_mid = 0
            if self.last_mid is not None:
                delta_mid = mid - self.last_mid
            state = self._discretize_state(self.getHoldings(self.symbol), delta_mid)

            mtm = self._mark_to_market(mid)
            inv = self.getHoldings(self.symbol)
            reward = self._compute_reward(mtm)
            self.cum_reward += reward

            action_idx = self._choose_action(state)
            if self.last_state is not None and self.last_action is not None:
                self._update_q(self.last_state, self.last_action, reward, state, action_idx)

            bid_offset, ask_offset = self.actions[action_idx]
            if bid_offset is None and ask_offset is None:
                self.cancelOrders()
            else:
                old_order_ids = list(self.orders.keys())
                self._place_quotes(mid, bid_offset, ask_offset)
                for oid in old_order_ids:
                    if oid in self.orders:
                        self.cancelOrder(self.orders[oid])

            self.last_delta_mid = delta_mid
            self._log_state(currentTime, mid, state, action_idx, reward)

            self.last_state = state
            self.last_action = action_idx
            self.last_mtm = mtm
            self.last_mid = mid
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def _place_quotes(self, mid, bid_offset, ask_offset):
        inv = self.getHoldings(self.symbol)
        if bid_offset is not None and inv < self.inventory_limit:
            bid_price = int(mid - bid_offset)
            self.placeLimitOrder(self.symbol, self.base_size, True, bid_price)
        if ask_offset is not None and inv > -self.inventory_limit:
            ask_price = int(mid + ask_offset)
            self.placeLimitOrder(self.symbol, self.base_size, False, ask_price)

    def _discretize_state(self, inventory, delta_mid):
        inv = int(np.clip(inventory, -self.inventory_clip, self.inventory_clip))
        dmid = int(np.clip(delta_mid, -self.delta_mid_clip, self.delta_mid_clip))
        inv_bin = inv // self.inventory_bin
        dmid_bin = dmid // self.delta_mid_bin
        return (inv_bin, dmid_bin)

    def _choose_action(self, state):
        self.step_count += 1
        eps = self.epsilon * (0.5 ** (self.step_count / self.epsilon_decay_steps))
        if self.random_state.rand() < eps:
            return self.random_state.randint(0, len(self.actions))
        qs = [self.q.get((state, a), 0.0) for a in range(len(self.actions))]
        return int(np.argmax(qs))

    def _update_q(self, state, action, reward, next_state, next_action):
        key = (state, action)
        current_q = self.q.get(key, 0.0)
        next_q = self.q.get((next_state, next_action), 0.0)
        target = reward + self.gamma * next_q
        self.q[key] = current_q + self.alpha * (target - current_q)

    def _compute_reward(self, mtm):
        return mtm - self.last_mtm

    def _mark_to_market(self, mid):
        if mid is None:
            mid = self.last_trade.get(self.symbol, 0) or 0
        return self.holdings['CASH'] + self.getHoldings(self.symbol) * mid

    def cancelOrders(self):
        for _, order in list(self.orders.items()):
            self.cancelOrder(order)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    def _log_state(self, currentTime, mid, state, action_idx, reward):
        inv_bin, dmid_bin = state
        qs = [self.q.get((state, a), 0.0) for a in range(len(self.actions))]
        q_max = max(qs) if qs else 0.0
        greedy_action = int(np.argmax(qs)) if qs else None
        self.logEvent('STATE', {
            'time': currentTime,
            'mid': mid,
            'inventory': self.getHoldings(self.symbol),
            'cash': self.holdings['CASH'],
            'mtm': self._mark_to_market(mid),
            'last_action': action_idx,
            'bid_offset': self.actions[action_idx][0],
            'ask_offset': self.actions[action_idx][1],
            'greedy_action': greedy_action,
            'inventory_bin': inv_bin,
            'delta_mid': self.last_delta_mid,
            'delta_mid_bin': dmid_bin,
            'reward': reward,
            'cum_reward': self.cum_reward,
            'q_max': q_max,
        })
