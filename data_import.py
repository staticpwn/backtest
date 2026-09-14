from mylogging import log_event, capture_error
import pandas as pd
import asyncio
from telegram_tools import *
import sys
import re
from features import *
from datetime import datetime, timezone

def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

async def importer(client, pair, interval, start=None, end=None, limit=None):
    
    if start != None:
        # start = pd.to_datetime(start, dayfirst=True)
        start = int(pd.to_datetime(start, dayfirst=True).tz_localize("UTC").timestamp() * 1000)
    if end != None:
        # end = pd.to_datetime(end, dayfirst=True)
        end = int(pd.to_datetime(end, dayfirst=True).tz_localize("UTC").timestamp() * 1000)
    
    attempts = 0
    while True:
        if attempts == 10:
            break
        attempts += 1
        try:
        
            if limit is None:
                if end != None:
                    raw = await asyncio.wait_for(client.get_historical_klines(pair, interval, start , end), 1000)
                else:
                    raw = await asyncio.wait_for(client.get_historical_klines(pair, interval, start), 1000)
            else:
                if end != None:
                    raw = await asyncio.wait_for(client.get_historical_klines(pair, interval, end_str = end, limit=limit), 1000)
                else:
                    if start != None:
                        raw = await asyncio.wait_for(client.get_historical_klines(pair, interval, start_str = start, limit=limit), 1000)
                    else:
                        raw = await asyncio.wait_for(client.get_historical_klines(pair, interval), 1000)
                
            raw_formatted = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "market_buy", "Taker_buy_quote_asset_volume", "ignore"])

            raw_formatted["open_time"] = pd.to_datetime(raw_formatted["open_time"], unit="ms")# + pd.Timedelta(hours=4)
            raw_formatted["close_time"] = pd.to_datetime(raw_formatted["close_time"], unit="ms")# + pd.Timedelta(hours=4)
            for col in raw_formatted.columns:
                if col != "open_time" and col != "close_time":
                    raw_formatted[col] = pd.to_numeric(raw_formatted[col])
        
        
            return raw_formatted.fillna(0)
        
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            if "semaphore" not in str(e):
                log_event(f"pair: {pair} - interval: {interval}.\n {str(e)}", "log.txt")
            capture_error()
            await asyncio.sleep(1) 
            # return None
            pass



def get_frame_import_interval(frame_name):
    
    numeric = re.findall(f"[0-9]+", frame_name)[0]
    alpha = re.findall(f"[a-z]+", frame_name)[0]
    return numeric + alpha[0]


def get_tasks(client, frame_interval, limit, start, end, source_list):

    tasks = []

    if limit is None:
        if end != None:
            if start is None:
                tasks = [importer(client, pair, frame_interval, limit=None, end=end) for pair in source_list]
            else:
                tasks = [importer(client, pair, frame_interval, limit=None, start=start, end=end) for pair in source_list]
        else:
            if start == None:
                print("when limit and end are None, start can't be None")
                return None
            tasks = [importer(client, pair, frame_interval, limit=None, start=start) for pair in source_list]
    else:
        if end != None:
            if start == None:
                tasks = [importer(client, pair, frame_interval, limit=limit, end=end) for pair in source_list]
            else:    
                tasks = [importer(client, pair, frame_interval, limit=limit, start=start, end=end) for pair in source_list]
        else:
            tasks = [importer(client, pair, frame_interval, limit=limit) for pair in source_list]

    
    return tasks

async def new_update_frames(client, dict_of_pairs: dict, 
                            target_pairs: list = None, frames: dict = None, 
                            batch_length = 30, wait_time_factor = 2, start=None, 
                            end=None, limit=1000, sanitize=True, verbose=True, raw=False):
    
    first_pair = list(dict_of_pairs.keys())[0]
    
    if frames is None:
        frames = {frame_name:get_frame_import_interval(frame_name) for frame_name in dict_of_pairs[first_pair]["dict_of_frames"]}
    
    if target_pairs is None:  
        batched_pairs = [list(dict_of_pairs.keys())[i:i+batch_length] for i in range(0, len(dict_of_pairs), batch_length)]
    else:
        batched_pairs = [target_pairs[i:i+batch_length] for i in range(0, len(target_pairs), batch_length)]
    
    if verbose:    
        print(f"total number of pairs {len(target_pairs) if target_pairs else len(dict_of_pairs)} split to {len(batched_pairs)} batches of max length: {batch_length}\nfor {','.join(list(frames.keys()))}")
    
    wait_time = wait_time_factor*len(frames)
    for batch_of_pairs in batched_pairs:
        test_pair = batch_of_pairs[-1]
        attempt_counter = 0
        while True:
        
            try:
                attempt_counter += 1
                for frame in frames:
                    source_list = batch_of_pairs #ticker_fiat

                    tasks = get_tasks(client, frames[frame], limit, start, end, source_list)
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    failed_pairs = []
                    for pair, res in zip(source_list, results):

                        if isinstance(res, Exception) or res is None:
                            dict_of_pairs[pair]["dict_of_frames"][frame] = None
                            failed_pairs.append(pair)
                            continue
                        try:
                            if not raw:
                                dict_of_pairs[pair]["dict_of_frames"][frame] = prepare_df(results[source_list.index(pair)])
                            else:
                                dict_of_pairs[pair]["dict_of_frames"][frame] = results[source_list.index(pair)]
                        except:
                            dict_of_pairs[pair]["dict_of_frames"][frame] = None
                            failed_pairs.append(pair)
                            print(f"pair {pair} failed format for frame {frame}")
                            capture_error("dlog.txt")
                            pass
                    

                    retry_count = 0
                    while len(failed_pairs) > 0:

                        retry_count+=1
                        if retry_count == 4:

                            break

                        print(f"retrying {len(failed_pairs)} failed tasks for frame {frame} for batch {batched_pairs.index(batch_of_pairs)+1}.")

                        retry_tasks = get_tasks(client, frames[frame], limit, start, end, failed_pairs)
                        retry_pairs = failed_pairs.copy()

                        failed_pairs = []

                        retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

                        for pair, res in zip(retry_pairs, retry_results):
 
                            if isinstance(res, Exception) or res is None:
                                dict_of_pairs[pair]["dict_of_frames"][frame] = None
                                failed_pairs.append(pair)
                                continue
                            try:
                                if not raw:
                                    dict_of_pairs[pair]["dict_of_frames"][frame] = prepare_df(retry_results[retry_pairs.index(pair)])
                                else:
                                    dict_of_pairs[pair]["dict_of_frames"][frame] = retry_results[retry_pairs.index(pair)]
                            except:
                                dict_of_pairs[pair]["dict_of_frames"][frame] = None
                                print(f"pair {pair} failed format for frame {frame}")
                                failed_pairs.append(pair)
                                capture_error("dlog.txt")
                                pass
                    

                if verbose:   

                    if len(failed_pairs) == 0:
                        print(f"successfully imported batch {batched_pairs.index(batch_of_pairs)+1}.")
                    else:
                        print(f"imported batch {batched_pairs.index(batch_of_pairs)+1} with {len(failed_pairs)} failed imports.")                  
                        print(f"failed pairs were: {','.join(list(set(failed_pairs)))}")  

                if batch_of_pairs != batched_pairs[-1]:
                    await asyncio.sleep(wait_time)
                attempt_counter = 0
                break
            
            except Exception as e:
                
                
                capture_error()
                
                if attempt_counter <= 5:
                    print(f"had a crash while importing for batch {batched_pairs.index(batch_of_pairs)+1}.\nretrying in {wait_time} seconds.")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"had a crash while importing for batch {batched_pairs.index(batch_of_pairs)+1}.\nexceeded retries. skipping.")
                    break
                
    failed_pairs = []
    if sanitize:
        for pair in [pair for pair in batch_of_pairs for batch_of_pairs in batched_pairs]:
            for frame in dict_of_pairs[pair]["dict_of_frames"]:
                if dict_of_pairs[pair]["dict_of_frames"][frame] is None:
                    failed_pairs.append(pair)
                    break

        if len(failed_pairs) > 0:
            print(f"Failed import for {len(set(failed_pairs))} pairs.\nFailed pairs were:\n{','.join(list(set(failed_pairs)))}")
            
            for pair in set(failed_pairs):
                
                    del dict_of_pairs[pair]
            print(f"Cleansing of {len(set(failed_pairs))} failed pairs complete.")
    
    if verbose:   
        print("Importing done.")


