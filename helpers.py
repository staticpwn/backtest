
import numpy as np
import pandas as pd

def cluster(data, maxgap, min_max_mean : str):
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][0]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
            
    mean_groups = [getattr(np, min_max_mean)(i) for i in groups]
    return mean_groups

from scipy.signal import find_peaks
from sklearn.cluster import KMeans

def level_finder(ohlc_df,levels_reactions_limit=12):

    df = ohlc_df
    
    peaks = find_peaks(df.high)[0] 
    troughs = find_peaks(-df.low)[0] 

    clusters = min(levels_reactions_limit, len(np.array([*df.high[peaks], *df.low[troughs]]).reshape(-1,1)))
    
    if clusters == 0:
        return [df.high.max(), df.low.min()]

    all_model = KMeans(n_clusters = clusters , random_state=0)
    all_model.fit(np.array([*df.high[peaks], *df.low[troughs]]).reshape(-1,1))

    all_levels = []
    for i in range(0,len(all_model.cluster_centers_)):
        all_levels.append(all_model.cluster_centers_[i][0])
    all_levels.sort()
    all_levels.extend([df.high.max(), df.low.min()])
    

    return sorted(list(set(all_levels)))


def simple_level_break(df, target, confirm=False):

    shift = 0
    if confirm:
        shift = 1
    return_series = pd.Series([False] * len(df), dtype=bool, index=df.index)
    if (type(target) == int) | (type(target) == float):
        confirm_series = confirm & (df.close > (target + 0.15*df.short_spread)) if confirm else True
        return_series = (df.shift(shift).close > (target + 0.15*df.shift(shift).short_spread)) & (df.shift(shift).open < target) & (df.shift(shift).low < target) & confirm_series
    elif type(target) == str:
        confirm_series = confirm & (df.close > (df[target] + 0.15*df.short_spread)) if confirm else True
        return_series = (df.shift(shift).close > (df[target] + 0.15*df.shift(shift).short_spread)) & (df.shift(shift).open < df[target]) & (df.shift(shift).low < df[target]) & confirm_series
        
    return return_series.values

def simple_level_test(df, target, confirm=False):

    shift = 0
    if confirm:
        shift = 1

    try:
        series = pd.Series([False] * len(df), dtype=bool, index=df.index)
        return_series = None
        if (type(target) == int) | (type(target) == float):
            return_series = (((df.low - target).abs() < 0.1*df.short_spread) & (((df[["close", "open"]].min(axis=1) - df.low) / (df.high - df[["close", "open"]].max(axis=1))) > 0)) |\
                ((df.close > target) & (df.open > target) & (df.low < target))
        elif type(target) == str:
            confirm_series = confirm & (df.close > (df[target] + 0.15*df.short_spread)) if confirm else True
            return_series =   ((df.shift(shift).close > df.shift(shift)[target]) &\
                                (df.shift(shift).open > df.shift(shift)[target]) &\
                                (df.shift(shift).low < df.shift(shift)[target])) &\
                                    confirm_series
            
        
        
        return_series[df[target] == 0] = False
        if return_series is None:
            return series.values
        
        if not return_series.any():
            return return_series.values
        # successful_clustered_values = cluster(df[return_series].index.to_list(), 5, "max")  
        
        # series[successful_clustered_values] = True
        return return_series.values
    except Exception as e:
        print(e)
        return None

def get_ranges(index_list:list):
    list_of_ranges = []
    for idx in index_list[:-1]:
        list_of_ranges.append((idx, index_list[index_list.index(idx)+1]))

    return list_of_ranges

def level_test_buys_with_last_broken(df, list_of_column_names, confirm_breaks=False, confirm_tests=False):
    
    return_dict = {}
    
    for column in list_of_column_names:
        
        bool_series = simple_level_break(df, column, confirm=confirm_breaks)
        empty_series = pd.Series(index=df.index,dtype=int)
        
        
        index_list = df[bool_series].index.to_list()
        if len(index_list) > 0:
            list_of_ranges = get_ranges(index_list)
            
            if len(list_of_ranges) > 0:
                for range_tuple in list_of_ranges:
                    empty_series[range_tuple[0]:range_tuple[1]] = range_tuple[0]
            
            
                empty_series[list_of_ranges[-1][-1]:] = list_of_ranges[-1][-1]
            else:
                empty_series[index_list[-1]] = index_list[-1]
                
        test_series = simple_level_test(df, column, confirm=confirm_tests)
        diff_series = ((df[column] - df["close"]) / df["short_spread"])
        if test_series is not None:
            return_dict["tested"] = test_series.astype(int)
            return_dict["last_broken"] = empty_series.index - empty_series
            return_dict["spreads_from_close"] = diff_series
            return_dict["is_broken"] = bool_series
    
    return return_dict

def get_simple_slope_2(series):

    try:
        return np.rad2deg(np.arctan((series[-1]- series[0]) / len(series)))
    except:
        return 0



def confirmed_bullish_divergence_entry(
    df: pd.DataFrame,
    osc_col: str = "rsi",
    left_bars: int = 3,
    right_bars: int = 3,
    oversold_threshold: float = 35.0,
    max_wait_bars: int = 12,
) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    osc = pd.to_numeric(df[osc_col], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)

    n = len(df)
    entry = np.zeros(n, dtype=bool)

    last_pivot = None
    active_reclaim_level = np.nan
    active_expiry = -1

    for t in range(left_bars + right_bars, n):
        pivot_idx = t - right_bars
        window = low[t - left_bars - right_bars : t + 1]
        pivot_low = low[pivot_idx]

        if np.isfinite(pivot_low) and np.all(np.isfinite(window)):
            if pivot_low == np.min(window) and np.sum(window == pivot_low) == 1:
                current_pivot = (pivot_idx, pivot_low, osc[pivot_idx])

                if last_pivot is not None:
                    p1_idx, p1_low, p1_osc = last_pivot
                    p2_idx, p2_low, p2_osc = current_pivot

                    bullish_div = (
                        p2_low < p1_low
                        and p2_osc > p1_osc
                        and min(p1_osc, p2_osc) <= oversold_threshold
                    )

                    if bullish_div:
                        active_reclaim_level = np.nanmax(high[p1_idx : p2_idx + 1])
                        active_expiry = t + max_wait_bars
                        

                last_pivot = current_pivot

        if np.isfinite(active_reclaim_level):
            if t <= active_expiry and close[t] > active_reclaim_level:
                entry[t] = True
                active_reclaim_level = np.nan
                active_expiry = -1
            elif t > active_expiry:
                active_reclaim_level = np.nan
                active_expiry = -1

    return pd.Series(entry, index=df.index, dtype="bool")

from binance import AsyncClient

async def get_spot_usdt_pairs(client: AsyncClient) -> list[str]:
    """
    Returns all actively tradable spot USDT pairs.
    Excludes:
        - futures instruments
        - non-USDT quote assets
        - delisted/suspended pairs
    """

    exchange_info = await client.get_exchange_info()

    pairs = [
        symbol["symbol"]
        for symbol in exchange_info["symbols"]
        if (
            symbol["status"] == "TRADING"
            and symbol["isSpotTradingAllowed"]
            and symbol["quoteAsset"] == "USDT"
            and symbol["baseAsset"] != "USDT"
        )
    ]

    pairs.sort()

    return pairs

from bisect import bisect_left

async def binance_slippage_capacity(
    client,
    pair: str,
    maximum_allowed_slippage: float,
    transaction_cash: float | list[float] | None,
) -> dict[str, object]:
    """
    transaction_cash is quote notional:
      > 0: buy cash to spend
      < 0: sell quote cash to receive
      None: estimated_slippage is None

    maximum_buy / maximum_sell are quote notional.
    maximum_buy_quantity / maximum_sell_quantity are base quantity.
    """
    INF = float("inf")

    max_slip = float(maximum_allowed_slippage)
    if max_slip != max_slip or max_slip < 0.0:
        raise ValueError("maximum_allowed_slippage must be a non-negative float")

    if hasattr(client, "get_order_book"):
        book = await client.get_order_book(symbol=pair, limit=1000)
    elif hasattr(client, "fetch_order_book"):
        book = await client.fetch_order_book(pair, limit=1000)
    else:
        raise TypeError("client must support get_order_book or fetch_order_book")

    if not isinstance(book, dict):
        raise TypeError("order book response must be a dict")

    def parse_depth(raw_levels):
        levels = []
        for level in raw_levels:
            try:
                price = float(level[0])
                qty = float(level[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if 0.0 < price < INF and 0.0 < qty < INF:
                levels.append((price, qty))
        return levels

    asks = parse_depth(book.get("asks") or [])
    bids = parse_depth(book.get("bids") or [])

    asks.sort(key=lambda x: x[0])
    bids.sort(key=lambda x: x[0], reverse=True)

    def build_cum(levels):
        cum_cash = []
        cum_qty = []
        total_cash = 0.0
        total_qty = 0.0
        for price, qty in levels:
            total_cash += price * qty
            total_qty += qty
            cum_cash.append(total_cash)
            cum_qty.append(total_qty)
        return cum_cash, cum_qty

    ask_cash, ask_qty = build_cum(asks)
    bid_cash, bid_qty = build_cum(bids)

    best_ask = asks[0][0] if asks else 0.0
    best_bid = bids[0][0] if bids else 0.0

    def tol(value: float) -> float:
        return 1e-12 * max(1.0, abs(value))

    def max_buy_quote_qty() -> tuple[float, float]:
        if not asks or best_ask <= 0.0:
            return 0.0, 0.0

        limit_price = best_ask * (1.0 + max_slip)
        prev_cash = 0.0
        prev_qty = 0.0

        for i, (price, qty) in enumerate(asks):
            cash = ask_cash[i]
            qty_cum = ask_qty[i]
            limit_cash = limit_price * qty_cum

            if cash <= limit_cash + tol(limit_cash):
                prev_cash, prev_qty = cash, qty_cum
                continue

            denom = price - limit_price
            if denom <= 0.0:
                prev_cash, prev_qty = cash, qty_cum
                continue

            numerator = limit_price * prev_qty - prev_cash
            if numerator < 0.0:
                numerator = 0.0

            partial = numerator / denom
            if partial < 0.0:
                partial = 0.0
            elif partial > qty:
                partial = qty

            return prev_cash + price * partial, prev_qty + partial

        return prev_cash, prev_qty

    def max_sell_quote_qty() -> tuple[float, float]:
        if not bids or best_bid <= 0.0:
            return 0.0, 0.0

        limit_price = best_bid * (1.0 - max_slip)
        prev_cash = 0.0
        prev_qty = 0.0

        for i, (price, qty) in enumerate(bids):
            cash = bid_cash[i]
            qty_cum = bid_qty[i]
            limit_cash = limit_price * qty_cum

            if cash >= limit_cash - tol(limit_cash):
                prev_cash, prev_qty = cash, qty_cum
                continue

            denom = limit_price - price
            if denom <= 0.0:
                prev_cash, prev_qty = cash, qty_cum
                continue

            numerator = prev_cash - limit_price * prev_qty
            if numerator < 0.0:
                numerator = 0.0

            partial = numerator / denom
            if partial < 0.0:
                partial = 0.0
            elif partial > qty:
                partial = qty

            return prev_cash + price * partial, prev_qty + partial

        return prev_cash, prev_qty

    def buy_slippage(cash: float) -> float:
        if cash <= 0.0:
            return 0.0
        if not asks or best_ask <= 0.0:
            return INF

        idx = bisect_left(ask_cash, cash)

        if idx == len(ask_cash):
            if ask_cash and cash <= ask_cash[-1] + tol(ask_cash[-1]):
                if ask_qty[-1] <= 0.0:
                    return INF
                slip = (ask_cash[-1] / ask_qty[-1]) / best_ask - 1.0
                return max(0.0, slip)
            return INF

        prev_cash = ask_cash[idx - 1] if idx else 0.0
        prev_qty = ask_qty[idx - 1] if idx else 0.0
        remaining = cash - prev_cash

        if remaining <= tol(cash):
            if prev_qty <= 0.0:
                return 0.0
            slip = (prev_cash / prev_qty) / best_ask - 1.0
            return max(0.0, slip)

        price = asks[idx][0]
        qty = asks[idx][1]
        fill_qty = remaining / price

        if fill_qty > qty:
            fill_qty = qty

        total_qty = prev_qty + fill_qty
        if total_qty <= 0.0:
            return INF

        slip = (cash / total_qty) / best_ask - 1.0
        return max(0.0, slip)

    def sell_slippage(cash: float) -> float:
        if cash <= 0.0:
            return 0.0
        if not bids or best_bid <= 0.0:
            return INF

        idx = bisect_left(bid_cash, cash)

        if idx == len(bid_cash):
            if bid_cash and cash <= bid_cash[-1] + tol(bid_cash[-1]):
                if bid_qty[-1] <= 0.0:
                    return INF
                slip = 1.0 - (bid_cash[-1] / bid_qty[-1]) / best_bid
                return max(0.0, slip)
            return INF

        prev_cash = bid_cash[idx - 1] if idx else 0.0
        prev_qty = bid_qty[idx - 1] if idx else 0.0
        remaining = cash - prev_cash

        if remaining <= tol(cash):
            if prev_qty <= 0.0:
                return 0.0
            slip = 1.0 - (prev_cash / prev_qty) / best_bid
            return max(0.0, slip)

        price = bids[idx][0]
        qty = bids[idx][1]
        fill_qty = remaining / price

        if fill_qty > qty:
            fill_qty = qty

        total_qty = prev_qty + fill_qty
        if total_qty <= 0.0:
            return INF

        slip = 1.0 - (cash / total_qty) / best_bid
        return max(0.0, slip)

    def estimate_one(cash: float) -> float:
        if cash > 0.0:
            return buy_slippage(cash)
        if cash < 0.0:
            return sell_slippage(-cash)
        return 0.0

    def validate_cash(value) -> float:
        cash = float(value)
        if cash != cash:
            raise ValueError("transaction_cash cannot contain NaN")
        return cash

    maximum_buy, maximum_buy_quantity = max_buy_quote_qty()
    maximum_sell, maximum_sell_quantity = max_sell_quote_qty()

    maximum_buy = max(0.0, maximum_buy)
    maximum_buy_quantity = max(0.0, maximum_buy_quantity)

    maximum_sell = max(0.0, maximum_sell)
    maximum_sell_quantity = max(0.0, maximum_sell_quantity)

    if transaction_cash is None:
        cash_out = None
        estimated_slippage = None
    elif isinstance(transaction_cash, (list, tuple)):
        cash_list = [validate_cash(item) for item in transaction_cash]
        cash_out = cash_list
        estimated_slippage = [estimate_one(item) for item in cash_list]
    else:
        cash_value = validate_cash(transaction_cash)
        cash_out = cash_value
        estimated_slippage = estimate_one(cash_value)

    return {
        "pair": pair,
        "transaction_cash": cash_out,
        "maximum_buy": maximum_buy,
        "maximum_buy_quantity": maximum_buy_quantity,
        "maximum_sell": maximum_sell,
        "maximum_sell_quantity": maximum_sell_quantity,
        "estimated_slippage": estimated_slippage,
    }