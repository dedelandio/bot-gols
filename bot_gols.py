"""
BOT DE ALERTAS DE GOLS NO TELEGRAM
----------------------------------

O que ele faz
=============
- Coleta partidas (via API ou CSV), calcula gols esperados com um modelo Poisson simples
  usando força de ataque/defesa por time e médias de liga (casa/fora).
- Estima probabilidades para mercados populares: Over 2.5, BTTS (Ambas Marcam), Gol no 1º Tempo (Over 0.5 HT).
- Compara com as odds (quando disponíveis) e envia SINAIS no Telegram se houver valor
  (probabilidade do modelo > probabilidade implícita + margem).
- Pode rodar em loop agendado (APScheduler) para enviar sinais automaticamente.

⚠️ Avisos importantes
=====================
- Não existe "acertividade garantida" em apostas. O código é educativo e não garante lucro.
- Respeite as leis locais e os termos das plataformas de apostas.

Como usar (passo a passo)
========================
1) Crie um bot no Telegram via @BotFather e anote o TOKEN.
2) Descubra seu CHAT_ID (p.ex., enviando uma msg ao bot e usando ferramentas como @userinfobot),
   ou use o ID de um canal onde o bot seja admin.
3) Se for usar API futebol (opcional), pegue um token (ex.: https://www.football-data.org/)
   e coloque na variável FOOTBALL_DATA_TOKEN.
4) Crie um arquivo .env na mesma pasta com:

   TELEGRAM_BOT_TOKEN=xxxxx:yyyyy
   TELEGRAM_CHAT_ID=123456789
   FOOTBALL_DATA_TOKEN=opcional
   LEAGUE_CODES=PL,BSA,PD,SA,BL1,FL1  # escolha suas ligas (códigos da football-data)

5) Instale dependências:
   pip install python-telegram-bot==20.7 pandas numpy requests python-dotenv APScheduler

6) Rode uma vez para testar:
   python bot_gols.py --once

7) Para rodar em loop (a cada 30 min por padrão):
   python bot_gols.py

Sem API? Use CSV
================
- Crie dois arquivos CSV (UTF-8):
  a) results.csv  (histórico para calibrar forças)
     columns: date,league,home,away,home_goals,away_goals
  b) fixtures.csv (próximos jogos + odds opcionais)
     columns: date,league,home,away,odds_over25,odds_btts,odds_1h_over05

- O bot lerá esses arquivos se FOOTBALL_DATA_TOKEN não estiver definido.

Sugestões de thresholds (ajuste ao seu gosto)
=============================================
- margin_value = 0.04  # o modelo deve superar a prob implícita em pelo menos 4 p.p.
- min_edge = 0.03      # mínimo de edge (prob_model * odds - 1 >= 3%)
- min_prob = 0.48      # envia apenas sinais com prob >= 48%

"""
from __future__ import annotations

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta, timezone

# ---------------------------- Utilidades -------------------------------------

def implied_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None or pd.isna(odds) or odds <= 1.0:
        return None
    return 1.0 / odds

def poisson_prob_over(total_lambda: float, line: float = 2.5) -> float:
    """Probabilidade de total de gols > line (ex.: 2.5) assumindo Poisson para gols casa+fora.
    Aproxima combinando duas Poisson? Simplesmente usamos a soma como Poisson(lambda_sum).
    """
    lam = total_lambda
    k = int(math.floor(line))
    cdf = sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k + 1))
    return 1.0 - cdf

def poisson_prob_btts(lam_home: float, lam_away: float) -> float:
    # P(ambas marcam) = 1 - P(casa=0) - P(fora=0) + P(casa=0, fora=0)
    p0h = math.exp(-lam_home)
    p0a = math.exp(-lam_away)
    return 1 - p0h - p0a + (p0h * p0a)

def poisson_prob_first_half_over05(lam_home: float, lam_away: float) -> float:
    # Aproxima metade dos gols esperados no 1º tempo
    lam_ht = 0.45 * (lam_home + lam_away)  # fator empírico (ajuste livre)
    return 1 - math.exp(-lam_ht)

# ---------------------------- Modelagem --------------------------------------

@dataclass
class TeamStrength:
    attack: float
    defense: float

@dataclass
class MatchSignal:
    date: str
    league: str
    home: str
    away: str
    market: str
    prob: float
    odds: Optional[float]
    edge: Optional[float]

class PoissonModel:
    def __init__(self, decay_half_life_days: int = 180, min_matches: int = 8):
        self.decay_half_life_days = decay_half_life_days
        self.min_matches = min_matches

    @staticmethod
    def _decay_weight(days: float, half_life: float) -> float:
        return 0.5 ** (days / half_life)

    def fit(self, results: pd.DataFrame) -> Dict[str, TeamStrength]:
        cols = {c.lower(): c for c in results.columns}
        for needed in ["date","league","home","away","home_goals","away_goals"]:
            assert needed in cols, f"Coluna faltando: {needed}"
        df = results.rename(columns={
            cols["date"]:"date", cols["league"]:"league", cols["home"]:"home", cols["away"]:"away",
            cols["home_goals"]:"home_goals", cols["away_goals"]:"away_goals"
        }).copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        mu_home = df["home_goals"].mean()
        mu_away = df["away_goals"].mean()
        today = df["date"].max() if not df.empty else pd.Timestamp.utcnow()
        df["days_ago"] = (today - df["date"]).dt.days.clip(lower=0)
        df["w"] = df["days_ago"].apply(lambda d: self._decay_weight(d, self.decay_half_life_days))

        teams = pd.unique(pd.concat([df["home"], df["away"]]))
        atk = {t: 1.0 for t in teams}
        dfn = {t: 1.0 for t in teams}

        for _ in range(12):
            for t in teams:
                home_rows = df[df["home"] == t]
                away_rows = df[df["away"] == t]
                w_sum = home_rows["w"].sum() + away_rows["w"].sum()
                if w_sum < self.min_matches:
                    atk[t] = 1.0
                    continue
                exp_home = (home_rows["w"] * (mu_home * dfn_series(home_rows["away"], dfn))).sum()
                exp_away = (away_rows["w"] * (mu_away * dfn_series(away_rows["home"], dfn))).sum()
                obs_home = (home_rows["w"] * home_rows["home_goals"]).sum()
                obs_away = (away_rows["w"] * away_rows["away_goals"]).sum()
                denom = exp_home + exp_away
                atk[t] = (obs_home + obs_away) / denom if denom > 0 else 1.0

            for t in teams:
                home_rows = df[df["home"] == t]
                away_rows = df[df["away"] == t]
                w_sum = home_rows["w"].sum() + away_rows["w"].sum()
                if w_sum < self.min_matches:
                    dfn[t] = 1.0
                    continue
                exp_home = (home_rows["w"] * (mu_home * atk_series(home_rows["away"], atk))).sum()
                exp_away = (away_rows["w"] * (mu_away * atk_series(away_rows["home"], atk))).sum()
                obs_home_conc = (home_rows["w"] * home_rows["away_goals"]).sum()
                obs_away_conc = (away_rows["w"] * away_rows["home_goals"]).sum()
                denom = exp_home + exp_away
                dfn[t] = (obs_home_conc + obs_away_conc) / denom if denom > 0 else 1.0

        return {t: TeamStrength(attack=atk[t], defense=dfn[t]) for t in teams}

    def predict_lambdas(self, strengths: Dict[str, TeamStrength], mu_home: float, mu_away: float,
                         home: str, away: str) -> Tuple[float,float]:
        th = strengths.get(home, TeamStrength(1.0,1.0))
        ta = strengths.get(away, TeamStrength(1.0,1.0))
        lam_home = mu_home * th.attack * ta.defense
        lam_away = mu_away * ta.attack * th.defense
        return float(lam_home), float(lam_away)


def atk_series(teams: pd.Series, atk: Dict[str, float]) -> pd.Series:
    return teams.map(lambda t: atk.get(t, 1.0))

def dfn_series(teams: pd.Series, dfn: Dict[str, float]) -> pd.Series:
    return teams.map(lambda t: dfn.get(t, 1.0))

# ---------------------------- Dados ------------------------------------------
def load_results_csv(path: str = "results.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date","league","home","away","home_goals","away_goals"])
    df = pd.read_csv(path)
    return df

def load_fixtures_csv(path: str = "fixtures.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date","league","home","away","odds_over25","odds_btts","odds_1h_over05"])
    return pd.read_csv(path)

FD_BASE = "https://api.football-data.org/v4"

def fd_headers(token: str) -> Dict[str,str]:
    return {"X-Auth-Token": token}

def fetch_fd_fixtures(leagues: List[str], token: str, days_ahead: int = 2) -> pd.DataFrame:
    rows = []
    until = datetime.utcnow() + timedelta(days=days_ahead)
    for code in leagues:
        url = f"{FD_BASE}/competitions/{code}/matches"
        params = {"status": "SCHEDULED", "dateFrom": datetime.utcnow().date().isoformat(), "dateTo": until.date().isoformat()}
        r = requests.get(url, headers=fd_headers(token), params=params, timeout=30)
        if r.status_code != 200:
            continue
        data = r.json()
        for m in data.get("matches", []):
            rows.append({
                "date": m.get("utcDate"),
                "league": code,
                "home": m.get("homeTeam",{}).get("name"),
                "away": m.get("awayTeam",{}).get("name"),
                "odds_over25": np.nan,
                "odds_btts": np.nan,
                "odds_1h_over05": np.nan,
            })
    if not rows:
        return pd.DataFrame(columns=["date","league","home","away","odds_over25","odds_btts","odds_1h_over05"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None)
    return df

def fetch_fd_results(leagues: List[str], token: str, days_back: int = 180) -> pd.DataFrame:
    rows = []
    since = datetime.utcnow() - timedelta(days=days_back)
    for code in leagues:
        url = f"{FD_BASE}/competitions/{code}/matches"
        params = {"status": "FINISHED", "dateFrom": since.date().isoformat(), "dateTo": datetime.utcnow().date().isoformat()}
        r = requests.get(url, headers=fd_headers(token), params=params, timeout=30)
        if r.status_code != 200:
            continue
        data = r.json()
        for m in data.get("matches", []):
            rows.append({
                "date": m.get("utcDate"),
                "league": code,
                "home": m.get("homeTeam",{}).get("name"),
                "away": m.get("awayTeam",{}).get("name"),
                "home_goals": m.get("score",{}).get("fullTime",{}).get("home", np.nan),
                "away_goals": m.get("score",{}).get("fullTime",{}).get("away", np.nan),
            })
    if not rows:
        return pd.DataFrame(columns=["date","league","home","away","home_goals","away_goals"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None)
    return df

# ---------------------------- Sinais -----------------------------------------
@dataclass
class MatchSignal:
    date: str
    league: str
    home: str
    away: str
    market: str
    prob: float
    odds: Optional[float]
    edge: Optional[float]

def build_signals(fixtures: pd.DataFrame, results: pd.DataFrame,
                  margin_value: float = 0.04, min_edge: float = 0.03, min_prob: float = 0.48) -> List[MatchSignal]:
    if fixtures.empty or results.empty:
        return []
    model = PoissonModel()
    strengths = model.fit(results)
    mu_home = results["home_goals"].mean()
    mu_away = results["away_goals"].mean()
    signals: List[MatchSignal] = []
    for _, row in fixtures.iterrows():
        lam_h, lam_a = model.predict_lambdas(strengths, mu_home, mu_away, row["home"], row["away"])
        p_over25 = poisson_prob_over(lam_h + lam_a, 2.5)
        p_btts = poisson_prob_btts(lam_h, lam_a)
        p_ht = poisson_prob_first_half_over05(lam_h, lam_a)

        candidates = [
            ("Over 2.5", p_over25, row.get("odds_over25", np.nan)),
            ("BTTS - Sim", p_btts, row.get("odds_btts", np.nan)),
            ("Gol no 1º Tempo (Over 0.5 HT)", p_ht, row.get("odds_1h_over05", np.nan)),
        ]
        for market, p, odds in candidates:
            imp = implied_prob(float(odds)) if not pd.isna(odds) else None
            edge = None
            send = False
            if imp is not None:
                edge = p * float(odds) - 1.0
                send = (p >= min_prob) and (p - imp >= margin_value) and (edge >= min_edge)
            else:
                send = (p >= max(min_prob, 0.55))

            if send:
                signals.append(MatchSignal(
                    date=str(row["date"])[:16], league=str(row["league"]), home=str(row["home"]), away=str(row["away"]),
                    market=market, prob=float(p), odds=float(odds) if not pd.isna(odds) else None,
                    edge=float(edge) if edge is not None else None
                ))
    return signals

# ---------------------------- Telegram ---------------------------------------
def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception as e:
        print("Erro Telegram:", e)

def format_signal(sig: MatchSignal) -> str:
    p = f"{sig.prob*100:.1f}%"
    odds = f" | Odds: {sig.odds:.2f}" if sig.odds else ""
    edge = f" | Edge: {sig.edge*100:.1f}%" if sig.edge is not None else ""
    return (f"<b>{sig.league}</b> | {sig.date}\n"
            f"<b>{sig.home}</b> x <b>{sig.away}</b>\n"
            f"<b>Mercado:</b> {sig.market}\n"
            f"<b>Prob:</b> {p}{odds}{edge}")

# ---------------------------- Main Loop --------------------------------------
def run_once() -> int:
    load_dotenv()
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    fd_token = os.getenv("FOOTBALL_DATA_TOKEN")
    league_codes = os.getenv("LEAGUE_CODES", "PL,BSA,PD,SA,BL1,FL1").split(",")

    if fd_token:
        fixtures = fetch_fd_fixtures(league_codes, fd_token)
        results = fetch_fd_results(league_codes, fd_token)
    else:
        fixtures = load_fixtures_csv()
        results = load_results_csv()

    signals = build_signals(fixtures, results)

    if not signals:
        print("Nenhum sinal elegível no momento.")
        return 0

    for s in signals:
        text = format_signal(s)
        print(text.replace("<","[").replace(">","]"))
        if tg_token and chat_id:
            send_telegram_message(tg_token, chat_id, text)
    return len(signals)


def schedule_loop():
    scheduler = BlockingScheduler(timezone="America/Fortaleza")  # ajustado para seu fuso

    @scheduler.scheduled_job('interval', minutes=30)
    def job():
        try:
            n = run_once()
            print(f"[{datetime.now().isoformat(timespec='seconds')}] Sinais enviados: {n}")
        except Exception as e:
            print("Erro no job:", e)

    print("Rodando a cada 30 minutos. Pressione Ctrl+C para sair.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', help='Executa apenas uma vez')
    args = parser.parse_args()
    if args.once:
        run_once()
    else:
        schedule_loop()
