"""
═══════════════════════════════════════════════════════════════════════════════
  INSTITUTIONAL GRADE STOCK RISK MONITOR v2.0
  With Options Flow Intelligence
  
  Six Pillars:
  1. Technical Analysis (30%)
  2. Options Flow (20%)
  3. Street Sentiment (15%)
  4. Institutional Flow (15%)
  5. Volatility Regime (10%)
  6. Event Risk (10%)
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

st.set_page_config(page_title="Institutional Grade Stock Monitor v2.0", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .metric-card { background-color: #1A1A2E; border-radius: 10px; padding: 15px; border: 1px solid #2D2D44; }
    .bullish { color: #00D4AA; } .bearish { color: #E74C3C; } .neutral { color: #FFEAA7; }
    .score-box { font-size: 36px; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

class SignalDirection(Enum): BULLISH = "Bullish"; NEUTRAL = "Neutral"; BEARISH = "Bearish"
class Confidence(Enum): LOW = "Low"; MEDIUM = "Medium"; HIGH = "High"
class TradeRisk(Enum): LOW = "Low"; MEDIUM = "Medium"; HIGH = "High"

@dataclass
class OptionsMetrics:
    call_oi: int; put_oi: int; call_volume: int; put_volume: int
    call_put_oi_ratio: float; call_put_vol_ratio: float
    total_oi: int; total_volume: int; implied_move: float
    avg_call_iv: float; avg_put_iv: float; iv_skew: float
    max_pain: float; unusual_activity_score: float
    smart_money_direction: str; smart_money_confidence: float

@dataclass
class OptionsSignal: score: float; signals: Dict; components: Dict

@dataclass
class CompositeScore:
    direction: SignalDirection; confidence: Confidence; trade_risk: TradeRisk
    total_score: float; technical_contrib: float; sentiment_contrib: float
    flow_contrib: float; options_contrib: float; volatility_contrib: float
    event_contrib: float; regime: str

class OptionsEngine:
    @staticmethod
    def analyze_chain(calls_df, puts_df, current_price, hist_oi=500000, hist_vol=200000):
        call_oi = int(calls_df['openInterest'].sum()) if 'openInterest' in calls_df.columns else 0
        put_oi = int(puts_df['openInterest'].sum()) if 'openInterest' in puts_df.columns else 0
        call_vol = int(calls_df['volume'].sum()) if 'volume' in calls_df.columns else 0
        put_vol = int(puts_df['volume'].sum()) if 'volume' in puts_df.columns else 0
        cp_oi = call_oi / put_oi if put_oi > 0 else 0
        cp_vol = call_vol / put_vol if put_vol > 0 else 0
        avg_civ = calls_df['impliedVolatility'].mean() if 'impliedVolatility' in calls_df.columns else 0.25
        avg_piv = puts_df['impliedVolatility'].mean() if 'impliedVolatility' in puts_df.columns else 0.30
        iv_skew = avg_piv - avg_civ
        max_pain = OptionsEngine.calc_max_pain(calls_df, puts_df)
        strikes = calls_df['strike'].values if len(calls_df) > 0 else np.array([current_price])
        atm_strike = float(strikes[np.argmin(np.abs(strikes - current_price))]) if len(strikes) > 0 else current_price
        atm_call = calls_df[calls_df['strike'] == atm_strike] if len(calls_df) > 0 else pd.DataFrame()
        atm_put = puts_df[puts_df['strike'] == atm_strike] if len(puts_df) > 0 else pd.DataFrame()
        c_price = atm_call['lastPrice'].iloc[0] if len(atm_call) > 0 else 0
        p_price = atm_put['lastPrice'].iloc[0] if len(atm_put) > 0 else 0
        implied_move = ((c_price + p_price) / current_price) * np.sqrt(252/30) * 100 if current_price > 0 else 0
        oi_spike = (call_oi + put_oi) / hist_oi if hist_oi > 0 else 1
        vol_spike = (call_vol + put_vol) / hist_vol if hist_vol > 0 else 1
        unusual = min(100, (oi_spike * 20 + vol_spike * 30) / 2)
        itm_calls = calls_df[calls_df['strike'] < current_price] if len(calls_df) > 0 else pd.DataFrame()
        otm_calls = calls_df[calls_df['strike'] > current_price] if len(calls_df) > 0 else pd.DataFrame()
        itm_puts = puts_df[puts_df['strike'] > current_price] if len(puts_df) > 0 else pd.DataFrame()
        otm_puts = puts_df[puts_df['strike'] < current_price] if len(puts_df) > 0 else pd.DataFrame()
        itm_co = itm_calls['openInterest'].sum() if len(itm_calls) > 0 else 0
        otm_co = otm_calls['openInterest'].sum() if len(otm_calls) > 0 else 0
        itm_po = itm_puts['openInterest'].sum() if len(itm_puts) > 0 else 0
        otm_po = otm_puts['openInterest'].sum() if len(otm_puts) > 0 else 0
        sm_call = itm_co / (itm_co + otm_co) if (itm_co + otm_co) > 0 else 0.5
        sm_put = itm_po / (itm_po + otm_po) if (itm_po + otm_po) > 0 else 0.5
        if cp_oi > 1.5 and sm_call > 0.6: direction, conf = "Bullish", min(1.0, (cp_oi - 1.5)/2 + sm_call * 0.5)
        elif cp_oi < 0.7 and sm_put > 0.6: direction, conf = "Bearish", min(1.0, (0.7 - cp_oi)/0.7 + sm_put * 0.5)
        else: direction, conf = "Neutral", 0.3
        return OptionsMetrics(call_oi, put_oi, call_vol, put_vol, round(cp_oi, 2), round(cp_vol, 2), call_oi + put_oi, call_vol + put_vol, round(implied_move, 2), round(avg_civ, 3), round(avg_piv, 3), round(iv_skew, 4), round(max_pain, 2), round(unusual, 1), direction, round(conf, 2))

    @staticmethod
    def calc_max_pain(calls_df, puts_df):
        if len(calls_df) == 0 or len(puts_df) == 0: return 0
        all_strikes = sorted(set(calls_df['strike'].tolist() + puts_df['strike'].tolist()))
        min_pain = float('inf'); pain_strike = all_strikes[len(all_strikes)//2]
        for strike in all_strikes:
            total = 0
            for _, c in calls_df.iterrows():
                if c['strike'] < strike and 'openInterest' in c: total += c['openInterest'] * (strike - c['strike'])
            for _, p in puts_df.iterrows():
                if p['strike'] > strike and 'openInterest' in p: total += p['openInterest'] * (p['strike'] - strike)
            if total < min_pain: min_pain, pain_strike = total, strike
        return pain_strike

    @staticmethod
    def score_signal(metrics, current_price):
        score = 0; signals = {}
        if metrics.call_put_oi_ratio > 2.0: cp_score, cp_sig = 40, "Strong Bullish"
        elif metrics.call_put_oi_ratio > 1.5: cp_score, cp_sig = 25, "Bullish"
        elif metrics.call_put_oi_ratio > 1.0: cp_score, cp_sig = 10, "Slight Bullish"
        elif metrics.call_put_oi_ratio > 0.7: cp_score, cp_sig = -10, "Slight Bearish"
        elif metrics.call_put_oi_ratio > 0.5: cp_score, cp_sig = -25, "Bearish"
        else: cp_score, cp_sig = -40, "Strong Bearish"
        if metrics.iv_skew > 0.05: iv_score, iv_sig = -20, "Elevated Put Demand"
        elif metrics.iv_skew < -0.02: iv_score, iv_sig = 15, "Elevated Call Demand"
        else: iv_score, iv_sig = 0, "Balanced IV"
        if metrics.unusual_activity_score > 70:
            ua_score = 15 if metrics.call_put_vol_ratio > 1.2 else -15
            ua_sig = "High Activity - Calls" if metrics.call_put_vol_ratio > 1.2 else "High Activity - Puts"
        else: ua_score, ua_sig = 0, "Normal"
        sm_map = {"Bullish": 20, "Neutral": 0, "Bearish": -20}
        sm_score = sm_map.get(metrics.smart_money_direction, 0) 