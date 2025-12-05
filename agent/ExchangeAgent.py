# The ExchangeAgent expects a numeric agent id, printable name, agent type, timestamp to open and close trading,
# a list of equity symbols for which it should create order books, a frequency at which to archive snapshots
# of its order books, a pipeline delay (in ns) for order activity, the exchange computation delay (in ns),
# the levels of order stream history to maintain per symbol (maintains all orders that led to the last N trades),
# whether to log all order activity to the agent log, and a random state object (already seeded) to use
# for stochasticity.
from agent.Agent import Agent
from Message import Message
from OrderBook import OrderBook

import datetime as dt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
pd.set_option('display.max_rows', 500)
from scipy.sparse import dok_matrix
from tqdm import tqdm

from copy import deepcopy


class ExchangeAgent(Agent):

  def __init__(self, id, name, type, mkt_open, mkt_close, symbols, book_freq='S', wide_book=False, pipeline_delay = 40000,
               computation_delay = 1, stream_history = 0, log_orders = False, random_state = None):

    super().__init__(id, name, type, random_state)

    # Do not request repeated wakeup calls.
    self.reschedule = False

    # Store this exchange's open and close times.
    self.mkt_open = mkt_open
    self.mkt_close = mkt_close

    # Right now, only the exchange agent has a parallel processing pipeline delay.  This is an additional
    # delay added only to order activity (placing orders, etc) and not simple inquiries (market operating
    # hours, etc).
    self.pipeline_delay = pipeline_delay

    # Computation delay is applied on every wakeup call or message received.
    self.computation_delay = computation_delay

    # The exchange maintains an order stream of all orders leading to the last L trades
    # to support certain agents from the auction literature (GD, HBL, etc).
    self.stream_history = stream_history

    # Log all order activity?
    self.log_orders = log_orders

    # Create an order book for each symbol.
    self.order_books = {}

    for symbol in symbols:
      self.order_books[symbol] = OrderBook(self, symbol)

    # At what frequency will we archive the order books for visualization and analysis?
    self.book_freq = book_freq

    # Store orderbook in wide format? ONLY WORKS with book_freq == 0
    self.wide_book = wide_book

  # The exchange agent overrides this to obtain a reference to an oracle.
  # This is needed to establish a "last trade price" at open (i.e. an opening
  # price) in case agents query last trade before any simulated trades are made.
  # This can probably go away once we code the opening cross auction.
  def kernelInitializing (self, kernel):
    super().kernelInitializing(kernel)

    self.oracle = self.kernel.oracle

    # Obtain opening prices (in integer cents).  These are not noisy right now.
    for symbol in self.order_books:
      try:
        self.order_books[symbol].last_trade = self.oracle.params['r_bar']
      except AttributeError as e:
        pass


  # The exchange agent overrides this to additionally log the full depth of its
  # order books for the entire day.
  def kernelTerminating (self):
    super().kernelTerminating()

    # If the oracle supports writing the fundamental value series for its
    # symbols, write them to disk.
    if hasattr(self.oracle, 'f_log'):
      for symbol in self.oracle.f_log:
        dfFund = pd.DataFrame(self.oracle.f_log[symbol])
        if not dfFund.empty:
          dfFund.set_index('FundamentalTime', inplace=True)
          self.writeLog(dfFund, filename='fundamental_{}'.format(symbol))
    if self.book_freq is None: return
    else:
      # Iterate over the order books controlled by this exchange.
      for symbol in self.order_books:
        start_time = dt.datetime.now()
        self.logOrderBookSnapshots(symbol)
        end_time = dt.datetime.now()
        print(f"[exchange] order book logging duration: {end_time - start_time}")
        print("[exchange] order book archival complete")

  def receiveMessage(self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    # Unless the intent of an experiment is to examine computational issues within an Exchange,
    # it will typically have either 1 ns delay (near instant but cannot process multiple orders
    # in the same atomic time unit) or 0 ns delay (can process any number of orders, always in
    # the atomic time unit in which they are received).  This is separate from, and additional
    # to, any parallel pipeline delay imposed for order book activity.

    # Is the exchange closed?  (This block only affects post-close, not pre-open.)
    if currentTime > self.mkt_close:
      # Most messages after close will receive a 'MKT_CLOSED' message in response.  A few things
      # might still be processed, like requests for final trade prices or such.
      if msg.body['msg'] in ['LIMIT_ORDER', 'MARKET_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER']:
        self.sendMessage(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))

        # Don't do any further processing on these messages!
        return
      elif 'QUERY' in msg.body['msg']:
        # Specifically do allow querying after market close, so agents can get the
        # final trade of the day as their "daily close" price for a symbol.
        pass
      else:
        self.sendMessage(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))

        # Don't do any further processing on these messages!
        return

    # Log order messages only if that option is configured.  Log all other messages.
    if msg.body['msg'] in ['LIMIT_ORDER', 'MARKET_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER']:
      if self.log_orders: self.logEvent(msg.body['msg'], msg.body['order'].to_dict())
    else:
      self.logEvent(msg.body['msg'], msg.body['sender'])

    # Handle all message types understood by this exchange.
    if msg.body['msg'] == "WHEN_MKT_OPEN":

      self.sendMessage(msg.body['sender'], Message({"msg": "WHEN_MKT_OPEN", "data": self.mkt_open}))
    elif msg.body['msg'] == "WHEN_MKT_CLOSE":

      self.sendMessage(msg.body['sender'], Message({"msg": "WHEN_MKT_CLOSE", "data": self.mkt_close}))
    elif msg.body['msg'] == "QUERY_LAST_TRADE":
      symbol = msg.body['symbol']
      if symbol not in self.order_books:
        return
      else:

        # Return the single last executed trade price (currently not volume) for the requested symbol.
        # This will return the average share price if multiple executions resulted from a single order.
        self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_LAST_TRADE", "symbol": symbol,
                                                      "data": self.order_books[symbol].last_trade,
                                                      "mkt_closed": True if currentTime > self.mkt_close else False}))
    elif msg.body['msg'] == "QUERY_SPREAD":
      symbol = msg.body['symbol']
      depth = msg.body['depth']
      if symbol not in self.order_books:
        return
      else:

        # Return the requested depth on both sides of the order book for the requested symbol.
        # Returns price levels and aggregated volume at each level (not individual orders).
        self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_SPREAD", "symbol": symbol, "depth": depth,
                                                      "bids": self.order_books[symbol].snapshot_bids(depth),
                                                      "asks": self.order_books[symbol].snapshot_asks(depth),
                                                      "data": self.order_books[symbol].last_trade,
                                                      "mkt_closed": True if currentTime > self.mkt_close else False,
                                                      "book": ''}))
    elif msg.body['msg'] == "QUERY_ORDER_STREAM":
      symbol = msg.body['symbol']
      length = msg.body['length']

      if symbol not in self.order_books:
        return
      else:

        # We return indices [1:length] inclusive because the agent will want "orders leading up to the last
        # L trades", and the items under index 0 are more recent than the last trade.
        self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_ORDER_STREAM", "symbol": symbol, "length": length,
                                                      "mkt_closed": True if currentTime > self.mkt_close else False,
                                                      "orders": self.order_books[symbol].history[1:length + 1]
                                                      }))
    elif msg.body['msg'] == 'QUERY_TRANSACTED_VOLUME':
      symbol = msg.body['symbol']
      lookback_period = msg.body['lookback_period']
      if symbol not in self.order_books:
        return
      else:
        self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_TRANSACTED_VOLUME", "symbol": symbol,
                                                      "transacted_volume": self.order_books[symbol].rolling_volume_sum(lookback_period),
                                                      "mkt_closed": True if currentTime > self.mkt_close else False
                                                      }))
    elif msg.body['msg'] == "LIMIT_ORDER":
      order = msg.body['order']
      if order.symbol not in self.order_books:
        return
      else:
        # Hand the order to the order book for processing.
        self.order_books[order.symbol].process_limit_submit(deepcopy(order))
    elif msg.body['msg'] == "MARKET_ORDER":
      order = msg.body['order']
      if order.symbol not in self.order_books:
        return
      else:
        # Hand the market order to the order book for processing.
        self.order_books[order.symbol].process_market_submit(deepcopy(order))
    elif msg.body['msg'] == "CANCEL_ORDER":
      # Note: this is somewhat open to abuse, as in theory agents could cancel other agents' orders.
      # An agent could also become confused if they receive a (partial) execution on an order they
      # then successfully cancel, but receive the cancel confirmation first.  Things to think about
      # for later...
      order = msg.body['order']
      if order.symbol not in self.order_books:
        return
      else:
        # Hand the order to the order book for processing.
        self.order_books[order.symbol].void_limit_entry(deepcopy(order))
    elif msg.body['msg'] == 'MODIFY_ORDER':
      # Replace an existing order with a modified order.  There could be some timing issues
      # here.  What if an order is partially executed, but the submitting agent has not
      # yet received the norification, and submits a modification to the quantity of the
      # (already partially executed) order?  I guess it is okay if we just think of this
      # as "delete and then add new" and make it the agent's problem if anything weird
      # happens.
      order = msg.body['order']
      new_order = msg.body['new_order']
      if order.symbol not in self.order_books:
        return
      else:
        self.order_books[order.symbol].adjust_limit_entry(deepcopy(order), deepcopy(new_order))

  def logOrderBookSnapshots(self, symbol):
    """
    Log full depth quotes (price, volume) from this order book at some pre-determined frequency. Here we are looking at
    the actual log for this order book (i.e. are there snapshots to export, independent of the requested frequency).
    """
    def book_log_to_df(book):
      """Build a sparse DataFrame of quote snapshots from an order book log."""
      quotes = sorted(list(book.quotes_seen))
      log_len = len(book.book_log)
      quote_idx_dict = {quote: idx for idx, quote in enumerate(quotes)}
      quotes_times = []

      # Construct sparse matrix, where rows are timesteps, columns are quotes and elements are volume.
      S = dok_matrix((log_len, len(quotes)), dtype=int)  # Dictionary Of Keys based sparse matrix.

      for i, row in enumerate(tqdm(book.book_log, desc="Processing orderbook log")):
        quotes_times.append(row['QuoteTime'])
        for quote, vol in row.items():
          if quote == "QuoteTime":
            continue
          S[i, quote_idx_dict[quote]] = vol

      S = S.tocsc()  # Convert this matrix to Compressed Sparse Column format for pandas to consume.
      df = pd.DataFrame.sparse.from_spmatrix(S, columns=quotes)
      df.insert(0, 'QuoteTime', quotes_times, allow_duplicates=True)
      return df

    def get_quote_range_iterator(s):
      """ Helper method for order book logging. Takes pandas Series and returns python range() from first to last
          element.
      """
      forbidden_values = [0, 19999900] # TODO: Put constant value in more sensible place!
      quotes = sorted(s)
      for val in forbidden_values:
        try: quotes.remove(val)
        except ValueError:
          pass
      return quotes

    book = self.order_books[symbol]

    if book.book_log:

      print(f"[exchange] archiving order book for {symbol} ...")
      dfLog = book_log_to_df(book)
      dfLog.set_index('QuoteTime', inplace=True)
      dfLog = dfLog[~dfLog.index.duplicated(keep='last')]
      dfLog.sort_index(inplace=True)

      if str(self.book_freq).isdigit() and int(self.book_freq) == 0:  # Save all possible information
        # Get the full range of quotes at the finest possible resolution.
        quotes = get_quote_range_iterator(dfLog.columns.unique())

        # Restructure the log to have multi-level rows of all possible pairs of time and quote
        # with volume as the only column.
        if not self.wide_book:
          filledIndex = pd.MultiIndex.from_product([dfLog.index, quotes], names=['time', 'quote'])
          dfLog = dfLog.stack()
          dfLog = dfLog.reindex(filledIndex)

        filename = f'ORDERBOOK_{symbol}_FULL'

      else:  # Sample at frequency self.book_freq
        # With multiple quotes in a nanosecond, use the last one, then resample to the requested freq.
        dfLog = dfLog.resample(self.book_freq).ffill()
        dfLog.sort_index(inplace=True)

        # Create a fully populated index at the desired frequency from market open to close.
        # Then project the logged data into this complete index.
        time_idx = pd.date_range(self.mkt_open, self.mkt_close, freq=self.book_freq, closed='right')
        dfLog = dfLog.reindex(time_idx, method='ffill')
        dfLog.sort_index(inplace=True)

        if not self.wide_book:
          dfLog = dfLog.stack()
          dfLog.sort_index(inplace=True)

          # Get the full range of quotes at the finest possible resolution.
          quotes = get_quote_range_iterator(dfLog.index.get_level_values(1).unique())

          # Restructure the log to have multi-level rows of all possible pairs of time and quote
          # with volume as the only column.
          filledIndex = pd.MultiIndex.from_product([time_idx, quotes], names=['time', 'quote'])
          dfLog = dfLog.reindex(filledIndex)

        filename = f'ORDERBOOK_{symbol}_FREQ_{self.book_freq}'

      # Final cleanup
      if not self.wide_book:
        df = pd.DataFrame(index=dfLog.index)
        df['Volume'] = dfLog
        try:
          df = df.astype(pd.SparseDtype(df['Volume'].dtype, fill_value=0))
        except Exception:
          pass
      else:
        df = dfLog
        df = df.reindex(sorted(df.columns), axis=1)

      # Archive the order book snapshots directly to a file named with the symbol, rather than
      # to the exchange agent log.
      self.writeLog(df, filename=filename)
      print(f"[exchange] archive finished for {symbol}")

  def sendMessage (self, recipientID, msg):
    # The ExchangeAgent automatically applies appropriate parallel processing pipeline delay
    # to those message types which require it.
    # TODO: probably organize the order types into categories once there are more, so we can
    # take action by category (e.g. ORDER-related messages) instead of enumerating all message
    # types to be affected.
    if msg.body['msg'] in ['ORDER_ACCEPTED', 'ORDER_CANCELLED', 'ORDER_EXECUTED']:
      # Messages that require order book modification (not simple queries) incur the additional
      # parallel processing delay as configured.
      self.kernel.dispatch(self.id, recipientID, msg, delay=self.pipeline_delay)
      if self.log_orders: self.logEvent(msg.body['msg'], msg.body['order'].to_dict())
    else:
      # Other message types incur only the currently-configured computation delay for this agent.
      self.kernel.dispatch(self.id, recipientID, msg)

  # Simple accessor methods for the market open and close times.
  def getMarketOpen(self):
    return self.__mkt_open

  def getMarketClose(self):
    return self.__mkt_close
