import os
import sys
import time
import pandas as pd
from telegram_tools import *

def log_event(event, path_to_log=None, dated=True):
    if path_to_log is None:
        path_to_log = "log.txt"
    if not os.path.exists(path_to_log):
        open(path_to_log, "w").close()
    
    with open(path_to_log, "a") as log:
        if dated:
            log.write(str(pd.to_datetime(int(time.time()), unit="s")+pd.Timedelta(4,"h")))
            log.write("  "+"-"*10+"  ")
        log.write(str(event))
        log.write("\n"+"-"*60+"\n")
        log.close()

def capture_error(path_to_log=None, notify=False, notification_bot = None):
    
    if path_to_log is None:
        path_to_log = "log.txt"
    if not os.path.exists(path_to_log):
        open(path_to_log, "w").close()
            
    traceback_template = '''Traceback (most recent call last):
    File "%(filename)s", line %(lineno)s, in %(name)s
    %(type)s: %(message)s\n''' # Skipping the "actual line" item
    
    exc_type, exc_value, exc_traceback = sys.exc_info() # most recent (if any) by default
    
    if "semaphore" in str(exc_value):
        return
    traceback_details = {
                         'filename': exc_traceback.tb_frame.f_code.co_filename,
                         'lineno'  : exc_traceback.tb_lineno,
                         'name'    : exc_traceback.tb_frame.f_code.co_name,
                         'type'    : exc_type.__name__,
                         'message' : str(exc_value), # or see traceback._some_str()
                        }

    del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
    # This still isn't "completely safe", though!
    # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
    # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
    with open(path_to_log, "a") as log:
        log.write(f"{pd.to_datetime(time.time(), unit='s') + pd.Timedelta(4,'h')}\n")
        log.write(traceback_template % traceback_details)
        
        log.write("\n"+"-"*60+"\n")
        log.close()
    
    if notify:
        notification_bot.send_message(MY_CHAT_ID,traceback_template % traceback_details)  