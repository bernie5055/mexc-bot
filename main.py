import hashlib
import hmac
import time
import os
import logging
import json
import sys
from datetime import datetime
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np

load_dotenv()

API_KEY    = os.getenv("MEXC_API_KEY", "")
SECRET_KEY = os.getenv("MEXC_SECRET_KEY", "")
BASE_URL   = "https://api.mexc.com"
SYMBOL     = "BTCUSDT"
INTERVAL   = "60m"
SLEEP_SEC  = 60

# Risk settings optimized for $50 account
RISK_PERCENT    = 0.20  # 20% per trade = ~$10 per trade
TAKE_PROFIT_PCT = 0.06  # 6% take profit = ~$3 profit per win
STOP_LOSS_PCT   = 0.03  # 3% stop loss = ~$1.50 max loss per trade

RSI_PERIOD = 14
RSI_BUY    = 35
RSI_SELL   = 65
EMA_FAST   = 50
EMA_SLOW   = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


def _sign(params):
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hmac.new(SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()


def _headers():
    return {"X-MEXC-APIKEY": API_KEY, "Content-Type": "application/json"}


def get_klines(symbol, interval, limit=200):
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    num_cols = len(data[0]) if data else 8
    if num_cols >= 12:
        columns = ["open_time","open","high","low","close","volume",
                   "close_time","quote_volume","trades",
                   "taker_buy_base","taker_buy_quote","ignore"]
    else:
        columns = ["open_time","open","high","low","close","volume",
                   "close_time","quote_volume"]
    df = pd.DataFrame(data, columns=columns[:num_cols])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def get_balance(asset="USDT"):
    ts = int(time.time() * 1000)
    params = {"timestamp": ts}
    params["signature"] = _sign(params)
    r = requests.get(f"{BASE_URL}/api/v3/account", params=params, headers=_headers(), timeout=10)
    r.raise_for_status()
    for b in r.json().get("balances", []):
        if b["asset"] == asset:
            return float(b["free"])
    return 0.0


def place_order(symbol, side, quantity):
    ts = int(time.time() * 1000)
    params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity, "timestamp": ts}
    params["signature"] = _sign(params)
    r = requests.post(f"{BASE_URL}/api/v3/order", params=params, headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def get_signals(df):
    df = df.copy()
    df["rsi"]      = calc_rsi(df["close"], RSI_PERIOD)
    df["ema_fast"] = calc_ema(df["close"], EMA_FAST)
    df["ema_slow"] = calc_ema(df["close"], EMA_SLOW)
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    rsi      = latest["rsi"]
    price    = latest["close"]
    ema_fast = latest["ema_fast"]
    ema_slow = latest["ema_slow"]
    buy_signal  = (prev["rsi"] < RSI_BUY and rsi >= RSI_BUY and price > ema_slow)
    sell_signal = ((prev["rsi"] < RSI_SELL and rsi >= RSI_SELL) or (price < ema_fast and rsi > 50))
    return {"price": price, "rsi": round(rsi, 2), "ema_fast": round(ema_fast, 2),
            "ema_slow": round(ema_slow, 2), "buy_signal": buy_signal, "sell_signal": sell_signal}


class Position:
    def __init__(self):
        self.in_trade = False
        self.entry_price = 0.0
        self.quantity = 0.0
        self.take_profit = 0.0
        self.stop_loss = 0.0

    def open(self, price, qty):
        self.in_trade = True
        self.entry_price = price
        self.quantity = qty
        self.take_profit = price * (1 + TAKE_PROFIT_PCT)
        self.stop_loss   = price * (1 - STOP_LOSS_PCT)
        log.info(f"POSITION OPENED | Entry: ${price:.2f} | Qty: {qty} | TP: ${self.take_profit:.2f} | SL: ${self.stop_loss:.2f}")

    def close(self, reason="signal"):
        log.info(f"POSITION CLOSED | Reason: {reason}")
        self.in_trade = False
        self.entry_price = 0.0
        self.quantity = 0.0

    def check_exits(self, current_price):
        if not self.in_trade:
            return None
        if current_price >= self.take_profit:
            return "take_profit"
        if current_price <= self.stop_loss:
            return "stop_loss"
        return None


def run_bot():
    log.info("=" * 55)
    log.info("  MEXC Trading Bot - RSI + EMA Strategy")
    log.info(f"  Symbol: {SYMBOL} | Timeframe: {INTERVAL}")
    log.info(f"  Risk: {RISK_PERCENT*100}% | TP: {TAKE_PROFIT_PCT*100}% | SL: {STOP_LOSS_PCT*100}%")
    log.info("=" * 55)

    if not API_KEY or not SECRET_KEY:
        log.error("API keys not set! Check your .env file.")
        return

    position = Position()

    while True:
        try:
            df      = get_klines(SYMBOL, INTERVAL)
            signals = get_signals(df)
            price   = signals["price"]
            log.info(f"[{datetime.now().strftime('%H:%M:%S')}] Price: ${price:.2f} | RSI: {signals['rsi']} | EMA50: ${signals['ema_fast']:.2f} | EMA200: ${signals['ema_slow']:.2f}")

            if position.in_trade:
                exit_reason = position.check_exits(price)
                if exit_reason:
                    order = place_order(SYMBOL, "SELL", position.quantity)
                    pnl = (price - position.entry_price) / position.entry_price * 100
                    log.info(f"Exit: {exit_reason} | PnL: {pnl:.2f}%")
                    position.close(exit_reason)
                elif signals["sell_signal"]:
                    order = place_order(SYMBOL, "SELL", position.quantity)
                    pnl = (price - position.entry_price) / position.entry_price * 100
                    log.info(f"SELL signal | PnL: {pnl:.2f}%")
                    position.close("strategy_signal")
            elif signals["buy_signal"]:
                balance = get_balance("USDT")
                log.info(f"USDT Balance: ${balance:.2f}")
                if balance < 10:
                    log.warning("Insufficient balance (< $10), skipping.")
                else:
                    qty = round((balance * RISK_PERCENT) / price, 6)
                    order = place_order(SYMBOL, "BUY", qty)
                    log.info(f"BUY order placed: {json.dumps(order)}")
                    position.open(price, qty)
            else:
                log.info("No signal - waiting...")

        except requests.exceptions.RequestException as e:
            log.error(f"Network error: {e}")
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)

        time.sleep(SLEEP_SEC)


# AUTO RESTART WRAPPER
if __name__ == "__main__":
    restart_count = 0
    while True:
        try:
            if restart_count > 0:
                log.info(f"*** AUTO RESTARTING BOT (restart #{restart_count}) ***")
            run_bot()
        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            sys.exit(0)
        except Exception as e:
            restart_count += 1
            log.error(f"BOT CRASHED: {e}")
            log.info("Restarting in 30 seconds...")
            time.sleep(30)
