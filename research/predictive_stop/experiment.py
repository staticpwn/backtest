"""Monthly walk-forward incremental-return stop selection; exact repository book."""
import json
import hashlib
import os
import sys
import time
import shutil
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import joblib
import backtest_engine as engine

FACTORS = np.array([0.5, 1.0, 1.5])
MARGIN = 0.0025  # 0.25 return percentage points, fixed before test results.
START = pd.Timestamp("2020-10-01")
END = pd.Timestamp("2022-01-01")


def config():
    return engine.EngineConfig(initial_cash=5020.0, fee_bps=10.0,
        max_net_worth_loss_ratio=0.05, max_slippage_allowed=0.004,
        max_number_of_positions=5, maximum_position_of_total_net=0.2,
        start_date="01/10/2020", end_date="31/12/2021", max_hold_bars_val=12,
        trailing_arm_spread_multiple=1.0, trailing_exit_spread_multiple=0.2,
        min_signal_hold_bars_override=8, replacement_score_margin=0.0,
        exposure_multiplier_col="__none__", slippage_multiplier=1.0,
        priority_1_strategies=["entry_break_max__exit_break_max", "entry_break_boll_upper__exit_break_boll_upper"],
        skip_entries_with_tp_below_close=True)


@njit(cache=True)
def label_paths(times, pairs, close, high, stop, tp, qvpuc, ratchet, spread,
                exit_signal, any_entry, is_test, factors):
    """Mirror _evaluate_and_close_position, including frozen initial stop/ratchet.

    Each candidate uses a fixed nominal 1,000 USDT (liquidity capped) so its
    label does not depend on portfolio equity. Full book keeps actual sizing.
    No artificial time horizon: unclosed alternatives remain censored.
    """
    T = close.shape[0]
    M = len(times)
    returns = np.full((M, 3), np.nan)
    ends = np.full((M, 3), T, dtype=np.int32)
    reasons = np.zeros((M, 3), dtype=np.int8)
    for j in range(M):
        t0, n = times[j], pairs[j]
        c0 = close[t0, n]
        entry_cash = min(1000.0, 0.004*qvpuc[t0,n])
        if entry_cash < 1.0:
            continue
        slip_entry = entry_cash/qvpuc[t0,n]
        amount = entry_cash/(c0*(1.0+slip_entry))
        for k in range(3):
            initial_stop = max(c0 - (c0-stop[t0,n])*factors[k], c0*0.001)
            stop_now = initial_stop
            peak = max(c0, high[t0,n])
            pct = max(ratchet[t0,n], 0.0) if np.isfinite(ratchet[t0,n]) else 0.0
            for t in range(t0+1,T):
                c, h = close[t,n], high[t,n]
                if not np.isfinite(c):
                    continue
                hold = t-t0
                time_dud = False
                time_gain = False
                if not any_entry[t,n] and hold >= 12:
                    time_dud = (c-c0)/c0 < 0.01
                    mean = np.mean(close[max(0,t-2):t+1,n])
                    time_gain = ((c-c0)/c0 >= 0.1) and abs((c-mean)/c) < 0.03
                trailing = False
                sp = spread[t,n]
                if np.isfinite(sp) and sp > 0 and not any_entry[t,n]:
                    trailing = (c-c0)/sp > 1.0 and (peak-c)/sp >= 0.2
                tp_exit = h > tp[t,n]
                stop_exit = c < stop_now and close[t-1,n] < stop_now
                if is_test and hold <= 12:
                    stop_exit = False
                signal_exit = exit_signal[t,n] and hold >= 8
                if tp_exit or trailing or time_dud or time_gain or stop_exit or signal_exit:
                    price = c
                    reason = 6  # strategy exit
                    if tp_exit:
                        reason = 1 if tp[t,n] > c0 else 7
                        if reason == 1:
                            price = tp[t,n]
                    elif trailing:
                        reason = 2
                    elif time_dud:
                        reason = 3
                    elif time_gain:
                        reason = 4
                    elif stop_exit:
                        reason = 5
                    value = amount*price
                    q = qvpuc[t,n]
                    slip = value/q if q > 0 else 0.004
                    exit_value = value*(1.0-slip)
                    returns[j,k] = (exit_value*0.999-entry_cash*1.001)/entry_cash
                    ends[j,k] = t
                    reasons[j,k] = reason
                    break
                peak = max(peak,c)
                if pct > 0 and peak > 0:
                    stop_now = max(stop_now,peak*(1.0-pct))
    return returns, ends, reasons


def candidates(payload):
    close, high = payload["prices"]["close"], payload["prices"]["high"]
    any_entry = np.zeros_like(close, dtype=np.bool_)
    for signals in payload["strategies"].values():
        any_entry |= signals["entry"]
    frames = []
    for sid, (name, signals) in enumerate(payload["strategies"].items()):
        aux = payload["aux"][name]
        stop, tp, q = [aux[k] for k in ["stop_price", "take_profit_price", "quote_vol_per_unit_change"]]
        mask = signals["entry"] & np.isfinite(close) & (stop > 0) & (stop < close) & (tp > close) & (q > 250)
        tt, nn = np.where(mask)
        print("Labelling", sid, name, len(tt), flush=True)
        ret, ends, reasons = label_paths(tt,nn,close,high,stop,tp,q,aux["trailing_stop_pct"],
            aux["short_spread"],signals["exit"],any_entry,"test" in name, FACTORS)
        f = pd.DataFrame(dict(t=tt,n=nn,sid=sid, strategy=name,
            time=np.asarray(payload["timeline"])[tt], pair=np.asarray(payload["pairs"])[nn],
            stop_distance=(close[tt,nn]-stop[tt,nn])/close[tt,nn],
            tp_distance=(tp[tt,nn]-close[tt,nn])/close[tt,nn],
            ratchet_pct=aux["trailing_stop_pct"][tt,nn]))
        for k in range(3):
            f["ret_"+str(k)] = ret[:,k]
            f["end_"+str(k)] = ends[:,k]
            f["reason_"+str(k)] = reasons[:,k]
        frames.append(f)
    out = pd.concat(frames,ignore_index=True).sort_values(["t","sid","n"]).reset_index(drop=True)
    features = pd.read_pickle(HERE / "features.pkl")
    out = out.merge(features, on=["time","pair"], how="left",validate="many_to_one")
    out.to_pickle(HERE / "candidates.pkl")
    return out


def parity_check(payload, data):
    """Compare the fast label simulator to the unmodified repository evaluator."""
    cfg = config()
    close, high = payload["prices"]["close"],payload["prices"]["high"]
    T,N = close.shape
    arrays = payload["strategies"]
    auxes = {name: engine._get_strategy_aux_arrays(payload["aux"],name,np.ones(T,dtype=bool),T,N,cfg)
             for name in arrays}
    zero = np.zeros_like(close)
    complete = data[data[["ret_0","ret_1","ret_2"]].notna().all(axis=1)]
    parts = [complete.sample(60, random_state=13)]
    for k in [0, 2]:
        changed = complete[(complete["ret_"+str(k)]-complete.ret_1).abs() > 1e-10]
        parts.append(changed.sample(min(30,len(changed)),random_state=13+k))
    test = pd.concat(parts).drop_duplicates(subset=["t","n","sid"])
    if len(test) < 120:
        remaining = complete.drop(index=test.index)
        test = pd.concat([test,remaining.sample(120-len(test),random_state=19)])
    checked = 0
    reasons = {1:"vol_take_profit",2:"trailing_stop",3:"time_stop",4:"profit_lock_in",
               5:"stop_loss",6:"exit_signal",7:"vol_take_profit_below_entry"}
    for row in test.itertuples():
        t,n,name = row.t,row.n,row.strategy
        aux = payload["aux"][name]
        c = close[t,n]
        cash = min(1000.0,0.004*aux["quote_vol_per_unit_change"][t,n])
        amount = cash/(c*(1+cash/aux["quote_vol_per_unit_change"][t,n]))
        for k,factor in enumerate(FACTORS):
            stop = max(c-(c-aux["stop_price"][t,n])*factor,c*0.001)
            pos = engine.PositionState(pair=row.pair,strategy=name,regime=None,entry_bar=t,
                entry_time=payload["timeline"].iloc[t],entry_price=c,initial_stop_price=stop,
                stop_price=stop,peak_price=max(c,high[t,n]),
                trailing_ratchet_pct=float(aux["trailing_stop_pct"][t,n]),max_hold_bars=0,
                vol_take_profit_threshold=0,min_signal_hold_bars=8,amount=amount,
                entry_value=cash,entry_fee=cash*0.001)
            expected_end = getattr(row,"end_"+str(k))
            trade = None
            for j in range(t+1,expected_end+1):
                trade = engine._evaluate_and_close_position(t=j,n=n,ts=payload["timeline"].iloc[j],
                    pair=row.pair,pos=pos,strategy_arrays=arrays,strategy_aux_arrays=auxes,
                    close_arr=close,high_arr=high,zero_arr=zero,fee_rate=0.001,cfg=cfg)
                if trade is not None:
                    break
            assert trade is not None and trade.exit_bar == expected_end,(row.Index,k,trade,expected_end)
            assert trade.close_reason == reasons[getattr(row,"reason_"+str(k))]
            np.testing.assert_allclose(trade.return_pct,getattr(row,"ret_"+str(k)),atol=1e-12)
            checked += 1
    (HERE / "parity.json").write_text(json.dumps(dict(checked_paths=checked,passed=True,
        sampling="60 random, up to 30 tight-changed, up to 30 wide-changed, padded to 120 entries")))
    print("Label parity passed",checked,flush=True)


def walk_forward(payload,data):
    exclusions = {"t","n","sid","strategy","time","pair"}
    columns = [c for c in data if c not in exclusions and not c.startswith(("ret_","end_","reason_"))]
    X = data[columns].replace([np.inf,-np.inf],np.nan).astype(float)
    for sid in range(len(payload["strategies"])):
        X["strategy_"+str(sid)] = (data.sid == sid).astype(float)
    pred = np.full((len(data),2),np.nan)
    mature = data[["end_0","end_1","end_2"]].max(axis=1).to_numpy()
    valid = data[["ret_0","ret_1","ret_2"]].notna().all(axis=1).to_numpy()
    months=[]
    models_dir=HERE/"models"
    models_dir.mkdir(exist_ok=True)
    for month in pd.date_range(START, END, freq="MS", inclusive="left"):
        next_month = month+pd.offsets.MonthBegin(1)
        cut=int(np.searchsorted(np.asarray(payload["timeline"]),month.to_datetime64()))
        train = valid & (data.time < month).to_numpy() & (data.time >= month-pd.DateOffset(months=24)).to_numpy() & (mature < cut)
        test = ((data.time >= month)&(data.time < next_month)).to_numpy()
        assert np.all(mature[train] < cut)
        assert train.sum() >= 1000
        for j,k in enumerate([0,2]):
            model = HistGradientBoostingRegressor(loss="squared_error",max_iter=150,
                max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=10.0,
                learning_rate=0.05,early_stopping=False,random_state=42)
            target=(data["ret_"+str(k)]-data.ret_1).to_numpy()
            with threadpool_limits(limits=1):
                model.fit(X.loc[train],target[train])
                pred[test,j]=model.predict(X.loc[test])
            joblib.dump(dict(model=model,features=X.columns.tolist(),month=str(month),
                             stop_factor=float(FACTORS[k])),models_dir/(month.strftime("%Y-%m")+"_"+str(k)+".joblib"))
        months.append(dict(month=str(month.date()),training_entries=int(train.sum()),test_entries=int(test.sum()),
                           latest_training_label_bar=int(mature[train].max()),prediction_start_bar=cut))
        print("WF", months[-1],flush=True)
    data["pred_tight_delta"] = pred[:,0]
    data["pred_wide_delta"] = pred[:,1]
    pd.DataFrame(months).to_csv(HERE/"walk_forward.csv",index=False)
    data.to_pickle(HERE/"predictions.pkl")
    return data


@contextmanager
def corrected_entry_fee_accounting():
    """Debit existing ledger fees to cash without changing exit/gate rules."""
    original_open = engine._open_position_from_candidate
    original_size = engine.compute_entry_cash
    audit = dict(entries_debited=0,total_entry_fees_debited=0.0,nonnegative_cash_checks=0)
    def affordable_size(**kwargs):
        amount = original_size(**kwargs)
        fee_rate = kwargs["cfg"].fee_bps / 10000.0
        return min(amount, kwargs["cash"]/(1.0+fee_rate))
    def open_with_fee(**kwargs):
        pos,cash = original_open(**kwargs)
        if pos is not None:
            cash -= pos.entry_fee
            np.testing.assert_allclose(kwargs["cash"]-cash,pos.entry_value+pos.entry_fee,rtol=1e-12,atol=1e-8)
            assert cash >= -1e-8,"Fee debit would borrow cash"
            audit["entries_debited"] += 1
            audit["total_entry_fees_debited"] += pos.entry_fee
            audit["nonnegative_cash_checks"] += 1
        return pos,cash
    engine.compute_entry_cash = affordable_size
    engine._open_position_from_candidate = open_with_fee
    try:
        yield audit
    finally:
        engine.compute_entry_cash = original_size
        engine._open_position_from_candidate = original_open


def run_books(payload,data,fee_corrected=False):
    originals={name: aux["stop_price"].copy() for name,aux in payload["aux"].items()}
    close=payload["prices"]["close"]
    rows=[];distributions=[]
    policies={"baseline":None,"fixed_tight_0.5":0.5,"fixed_wide_1.5":1.5,
              "ml_widen_only": "wide", "ml_tighten_only":"tight", "ml_both":"both"}
    evaluation=data[(data.time >= START)&(data.time < END)].copy()
    diagnostics={}
    complete=evaluation[["ret_0","ret_1","ret_2"]].notna().all(axis=1)
    for k,label in [(0,"tight"),(2,"wide")]:
        delta=evaluation.loc[complete,"ret_"+str(k)]-evaluation.loc[complete,"ret_1"]
        diagnostics[label]=dict(complete_candidates=int(complete.sum()),
            changed_outcome_pct=float((delta.abs()>1e-10).mean()*100),
            mean_delta_pp=float(delta.mean()*100),
            improved_pct=float((delta>1e-10).mean()*100),
            worsened_pct=float((delta < -1e-10).mean()*100))
    (HERE/"label_diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    def book_digest():
        digest=hashlib.sha256()
        for name,signals in payload["strategies"].items():
            for array in signals.values():
                digest.update(array.tobytes())
            for key,array in payload["aux"][name].items():
                if key != "stop_price":
                    digest.update(array.tobytes())
        return digest.hexdigest()
    original_digest=book_digest()
    decisions=[]
    for arm,policy in policies.items():
        for name,stop in originals.items():
            payload["aux"][name]["stop_price"]=stop.copy()
        choices=np.ones(len(evaluation),dtype=np.int8)
        if isinstance(policy,float):
            choices[:]=0 if policy < 1 else 2
        elif isinstance(policy,str):
            tight=evaluation.pred_tight_delta.to_numpy()
            wide=evaluation.pred_wide_delta.to_numpy()
            if policy == "wide":
                choices[wide>MARGIN]=2
            elif policy == "tight":
                choices[tight>MARGIN]=0
            else:
                choices[(tight>MARGIN)&(tight>=wide)]=0
                choices[(wide>MARGIN)&(wide>tight)]=2
        for sid,name in enumerate(payload["strategies"]):
            mask=evaluation.sid.to_numpy()==sid
            subset=evaluation.loc[mask]
            tt,nn=subset.t.to_numpy(),subset.n.to_numpy()
            stop=originals[name]
            payload["aux"][name]["stop_price"][tt,nn]=np.maximum(
                close[tt,nn]-(close[tt,nn]-stop[tt,nn])*FACTORS[choices[mask]],close[tt,nn]*0.001)
        chosen=evaluation[["time","pair","strategy"]].copy()
        chosen["stop_factor"]=FACTORS[choices]
        chosen["arm"]=arm
        decisions.append(chosen)
        print("Book",arm,flush=True)
        result=engine.run_portfolio_backtest(payload,cfg=config())["portfolio"]
        trades=result["trade_frame"]
        trades.to_csv(HERE/(arm+"_trades.csv"),index=False)
        curves=pd.DataFrame({k:result[k] for k in ["equity_curve","cash_curve","open_positions_curve","closed_trades_curve"]})
        curves["time"]=payload["timeline"][(payload["timeline"]>=START)&(payload["timeline"]<END)].to_numpy()
        curves.to_csv(HERE/(arm+"_curves.csv"),index=False)
        m=result["metrics"]
        (HERE/(arm+"_metrics.json")).write_text(json.dumps(m,default=str,indent=2))
        # Current repository ledger stores fractional net returns.
        expectancy=float(trades.return_pct.mean()*100)
        winrate=float((trades.pnl>0).mean()*100)
        eq=curves.equity_curve.to_numpy()
        row=dict(arm=arm,net_multiple=float(eq[-1]/config().initial_cash),
            trades=len(trades),expectancy_pct=expectancy,win_rate_pct=winrate,
            max_drawdown_pct=float(np.min(eq/np.maximum.accumulate(eq)-1)*100),
            final_equity=float(eq[-1]),open_positions_at_end=int(curves.open_positions_curve.iloc[-1]),
            tight_candidate_pct=float((choices==0).mean()*100),wide_candidate_pct=float((choices==2).mean()*100))
        closed=trades.merge(
            chosen.rename(columns={"time":"entry_time"}),on=["pair","strategy","entry_time"],how="left",validate="many_to_one")
        row["tight_executed_pct"]=float((closed.stop_factor==0.5).mean()*100)
        row["wide_executed_pct"]=float((closed.stop_factor==1.5).mean()*100)
        rows.append(row)
        counts=trades.close_reason.value_counts()
        for reason,count in counts.items():
            distributions.append(dict(arm=arm,reason=reason,count=int(count),pct=float(count/len(trades)*100)))
        print(row,flush=True)
    summary=pd.DataFrame(rows)
    for c in ["expectancy_pct","win_rate_pct"]:
        summary["delta_"+c+"_pp"]=summary[c]-summary.loc[0,c]
    summary["relative_multiple_change_pct"]=(summary.net_multiple/summary.loc[0,"net_multiple"]-1)*100
    summary.to_csv(HERE/"summary.csv",index=False)
    assert book_digest()==original_digest,"A non-stop book input was modified"
    (HERE/"book_integrity.json").write_text(json.dumps(dict(non_stop_inputs_unchanged=True,
        sha256=original_digest),indent=2))
    pd.DataFrame(distributions).to_csv(HERE/"close_reasons.csv",index=False)
    pd.concat(decisions,ignore_index=True).to_csv(HERE/"decisions.csv",index=False)
    (HERE/"experiment_config.json").write_text(json.dumps(dict(engine=asdict(config()),stop_factors=FACTORS.tolist(),
        margin_return=MARGIN,model="HistGradientBoostingRegressor",walk_forward="monthly, trailing <=24 months",
        entry_fee_cash_debit=fee_corrected,
        label_notional_usdt=1000,training_label_requirement="All three policy exits strictly before prediction month",
        implementation_python=sys.version),indent=2,default=str))
    for name,stop in originals.items():
        payload["aux"][name]["stop_price"]=stop
    print(summary.to_string(index=False),flush=True)


def main():
    payload=pd.read_pickle(HERE/"payload.pkl")
    data=pd.read_pickle(HERE/"candidates.pkl") if (HERE/"candidates.pkl").exists() else candidates(payload)
    parity_check(payload,data)
    data=pd.read_pickle(HERE/"predictions.pkl") if (HERE/"predictions.pkl").exists() else walk_forward(payload,data)
    if "--fee-corrected-only" not in sys.argv:
        run_books(payload,data)
    for name in ["summary.csv","close_reasons.csv","experiment_config.json"]:
        source=HERE/name
        if source.exists():
            shutil.copyfile(source,HERE/("original_engine_"+name))
    with corrected_entry_fee_accounting() as audit:
        run_books(payload,data,fee_corrected=True)
    (HERE/"fee_accounting_audit.json").write_text(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
