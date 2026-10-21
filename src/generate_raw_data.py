"""
Generate the RAW (messy) session export used in this project.

Scenario
--------
The client (name withheld) is a mid-sized Brazilian online retailer
selling fashion and home goods, headquartered in Sao Paulo. Their analytics
team exported one row per website session for Oct-2025 .. Sep-2026.

All data is SYNTHETIC. It is generated so that realistic behavioural
patterns exist (e.g. cart adds, installment-page views and Black Friday drive
purchases) and it is intentionally DIRTY (mixed casing, Portuguese/English
labels, mixed date formats, missing values, duplicates, bot traffic, negative
durations) so the cleaning step is a real exercise.

Usage:  python src/generate_raw_data.py
Output: data/raw/sessions_raw.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_SESSIONS = 15_873          # deliberately not a round number
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "sessions_raw.csv"

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- reference
STATES = {
    "SP": 0.31, "RJ": 0.12, "MG": 0.11, "RS": 0.07, "PR": 0.07, "SC": 0.05,
    "BA": 0.05, "DF": 0.03, "GO": 0.03, "PE": 0.03, "CE": 0.03, "ES": 0.02,
    "PA": 0.02, "AM": 0.02, "MT": 0.02, "MS": 0.01, "RN": 0.01,
}
STATE_NAMES = {"SP": "São Paulo", "RJ": "Rio de Janeiro", "MG": "Minas Gerais",
               "RS": "Rio Grande do Sul", "BA": "Bahia", "PR": "Paraná"}
# South/Southeast get cheaper, faster freight -> converts slightly better
STATE_EFFECT = {s: (0.15 if s in {"SP", "RJ", "MG", "PR", "SC", "RS", "ES"} else -0.25)
                for s in STATES}

SOURCES = {
    "Google Orgânico": (0.24, 0.10), "Google Ads": (0.17, 0.25),
    "Instagram": (0.18, -0.15), "Facebook": (0.07, -0.30),
    "TikTok": (0.06, -0.45), "WhatsApp": (0.08, 0.45),
    "E-mail": (0.06, 0.55), "Direto": (0.10, 0.30), "Afiliados": (0.04, 0.05),
}
DEVICES = {"Mobile": (0.68, -0.20), "Desktop": (0.27, 0.35), "Tablet": (0.05, 0.0)}

# Brazilian retail calendar (start, end, effect, label)
EVENTS = [
    ("2025-11-24", "2025-12-01", 0.95, "Black Friday"),
    ("2025-12-10", "2025-12-23", 0.55, "Natal"),
    ("2026-02-14", "2026-02-17", -0.40, "Carnaval"),
    ("2026-03-09", "2026-03-15", 0.40, "Semana do Consumidor"),
    ("2026-05-01", "2026-05-10", 0.50, "Dia das Mães"),
    ("2026-06-05", "2026-06-12", 0.40, "Dia dos Namorados"),
]


def pick(d, n):
    keys = list(d)
    p = np.array([v[0] if isinstance(v, tuple) else v for v in d.values()], float)
    return rng.choice(keys, size=n, p=p / p.sum())


def main():
    n = N_SESSIONS

    # ---- timestamps: more traffic in Nov/Dec, evenings & lunch peaks
    days = pd.date_range("2025-10-01", "2026-09-30", freq="D")
    day_w = np.ones(len(days))
    for s, e, eff, _ in EVENTS:
        m = (days >= s) & (days <= e)
        day_w[m] *= 1.8 if eff > 0 else 0.8
    day = rng.choice(days, size=n, p=day_w / day_w.sum())
    hour_w = np.array([2, 1, 1, 1, 1, 2, 3, 5, 6, 7, 7, 8, 10, 9, 7, 7, 7, 8, 9, 11, 12, 12, 9, 5], float)
    hour = rng.choice(24, size=n, p=hour_w / hour_w.sum())
    ts = pd.to_datetime(day) + pd.to_timedelta(hour, "h") + pd.to_timedelta(rng.integers(0, 3600, n), "s")

    event_eff = np.zeros(n)
    for s, e, eff, _ in EVENTS:
        m = (ts >= s) & (ts <= pd.Timestamp(e) + pd.Timedelta(days=1))
        event_eff[m] = eff

    state = pick(STATES, n)
    source = pick(SOURCES, n)
    device = pick(DEVICES, n)
    returning = rng.random(n) < np.where(np.isin(source, ["E-mail", "Direto", "WhatsApp"]), 0.65, 0.32)

    # ---- latent shopping intent drives behaviour
    intent = (rng.normal(0, 1, n) + 0.45 * returning
              + np.vectorize(lambda s: SOURCES[s][1])(source)
              + np.vectorize(lambda d: DEVICES[d][1])(device) * 0.5)

    product_pages = rng.poisson(np.exp(1.2 + 0.45 * intent).clip(0.3, 40))
    category_pages = rng.poisson(np.exp(0.9 + 0.15 * intent).clip(0.3, 15))
    info_pages = rng.poisson(0.6 + 0.3 * (intent > 0.5), n)
    pages_viewed = 1 + product_pages + category_pages + info_pages
    searches = rng.poisson(np.exp(-0.3 + 0.35 * intent).clip(0.05, 10))

    sec_per_page = rng.lognormal(3.6, 0.55, n)                 # ~ 37s median
    session_duration = (pages_viewed * sec_per_page * np.where(device == "Mobile", 0.85, 1.1)).round()
    product_duration = (session_duration * rng.beta(2 + intent.clip(-1.5, 3), 3)).round()

    p_cart = 1 / (1 + np.exp(-(-1.3 + 1.1 * intent + 0.05 * product_pages)))
    add_to_cart = np.where(rng.random(n) < p_cart, rng.poisson(1.0, n) + 1, 0)
    wishlist = rng.poisson(np.exp(-1.8 + 0.4 * intent).clip(0, 5))

    viewed_freight = rng.random(n) < 1 / (1 + np.exp(-(-1.0 + 0.9 * intent + 0.8 * (add_to_cart > 0))))
    viewed_install = rng.random(n) < 1 / (1 + np.exp(-(-1.4 + 0.7 * intent + 0.6 * (add_to_cart > 0))))
    coupon_applied = rng.random(n) < np.where(add_to_cart > 0, 0.18, 0.02) * (1 + (event_eff > 0))

    bounce_rate = (rng.beta(2, 9, n) * np.exp(-0.25 * intent) + 0.4 * (pages_viewed <= 2)).clip(0, 1).round(4)
    exit_rate = (bounce_rate * 0.6 + rng.beta(2, 12, n) + 0.05).clip(0, 1).round(4)

    # avg value (R$) of the pages seen, like GA "page value"; mostly 0
    page_value = np.where(rng.random(n) < 1 / (1 + np.exp(-(-1.6 + 1.0 * intent))),
                          rng.gamma(2.0, 22 + 10 * intent.clip(0, 3)), 0.0).round(2)

    cart_value = np.where(add_to_cart > 0, rng.gamma(2.2, 85, n) * add_to_cart, 0).round(2)

    # ---- purchase outcome
    logit = (-4.45 + 0.55 * intent
             + 1.55 * (add_to_cart > 0) + 0.10 * np.minimum(add_to_cart, 5)
             + 0.030 * np.minimum(page_value, 150)
             + 0.55 * viewed_freight + 0.60 * viewed_install + 0.70 * coupon_applied
             - 1.2 * bounce_rate
             + np.vectorize(STATE_EFFECT.get)(state)
             + event_eff + 0.25 * returning
             + 0.004 * np.minimum(product_duration, 600) / 6)
    purchased = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame({
        "session_id": [f"CE-{x:07d}" for x in rng.choice(9_000_000, n, replace=False) + 1_000_000],
        "session_start": ts,
        "customer_state": state,
        "device_type": device,
        "traffic_source": source,
        "visitor_type": np.where(returning, "Recorrente", "Novo"),
        "pages_viewed": pages_viewed,
        "category_pages": category_pages,
        "product_pages": product_pages,
        "searches": searches,
        "session_duration_sec": session_duration,
        "product_page_duration_sec": product_duration,
        "add_to_cart_count": add_to_cart,
        "wishlist_adds": wishlist,
        "cart_value_brl": cart_value,
        "viewed_freight_calculator": viewed_freight,
        "viewed_installment_options": viewed_install,
        "coupon_applied": coupon_applied,
        "bounce_rate": bounce_rate,
        "exit_rate": exit_rate,
        "page_value_brl": page_value,
        "purchased": purchased,
    })

    df = make_it_messy(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Saved {len(df):,} rows -> {OUT.relative_to(ROOT)}")


def make_it_messy(df):
    n = len(df)
    df = df.copy()

    # 1) mixed date formats (BR dd/mm/yyyy and ISO)
    iso = df["session_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    br = df["session_start"].dt.strftime("%d/%m/%Y %H:%M")
    df["session_start"] = np.where(rng.random(n) < 0.35, br, iso)

    # 2) inconsistent state labels
    st = df["customer_state"].astype(object)
    r = rng.random(n)
    st = np.where(r < 0.06, pd.Series(st).str.lower(), st)
    st = np.where((r >= 0.06) & (r < 0.09), pd.Series(st).map(lambda s: STATE_NAMES.get(s, s)), st)
    st = np.where((r >= 0.09) & (r < 0.11), pd.Series(st).map(lambda s: f" {s} "), st)
    df["customer_state"] = st

    # 3) device labels from two tracking tags (PT/EN, casing)
    dmap = {"Mobile": ["Mobile", "mobile", "Celular", "MOBILE"],
            "Desktop": ["Desktop", "desktop", "Computador"],
            "Tablet": ["Tablet", "tablet"]}
    df["device_type"] = [rng.choice(dmap[d], p=None) if rng.random() < 0.3 else d for d in df["device_type"]]

    # 4) booleans exported in several styles
    for col in ["viewed_freight_calculator", "viewed_installment_options", "coupon_applied"]:
        style = rng.random(n)
        v = df[col].to_numpy()
        df[col] = np.select([style < 0.5, style < 0.8],
                            [np.where(v, "sim", "não"), np.where(v, "1", "0")],
                            np.where(v, "True", "False"))
    df["purchased"] = np.where(rng.random(n) < 0.8, df["purchased"].map({True: "sim", False: "não"}),
                               df["purchased"].astype(int).astype(str))

    # 5) money as BR text "R$ 1.234,56" for part of the rows
    def brl(x):
        return "R$ " + f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    m = rng.random(n) < 0.25
    df["cart_value_brl"] = df["cart_value_brl"].astype(object)
    df.loc[m, "cart_value_brl"] = df.loc[m, "cart_value_brl"].map(brl)

    # 6) missing values
    for col, frac in [("customer_state", 0.021), ("traffic_source", 0.014), ("device_type", 0.008),
                      ("session_duration_sec", 0.031), ("product_page_duration_sec", 0.027),
                      ("exit_rate", 0.012), ("visitor_type", 0.006)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 7) tracking glitches: negative durations
    idx = rng.choice(n, 37, replace=False)
    df.loc[idx, "session_duration_sec"] = -df.loc[idx, "session_duration_sec"].abs()

    # 8) bot / crawler sessions (huge page counts in seconds, never buy)
    bots = df.sample(118, random_state=SEED).copy()
    bots["pages_viewed"] = rng.integers(180, 900, len(bots))
    bots["product_pages"] = (bots["pages_viewed"] * 0.8).astype(int)
    bots["session_duration_sec"] = rng.integers(3, 40, len(bots))
    bots["add_to_cart_count"] = 0
    bots["purchased"] = "não"
    bots["session_id"] = [f"CE-BOT{x:05d}" for x in range(len(bots))]
    df = pd.concat([df, bots], ignore_index=True)

    # 9) exact duplicate rows from a double export
    dups = df.sample(211, random_state=7)
    df = pd.concat([df, dups], ignore_index=True)

    return df.sample(frac=1, random_state=11).reset_index(drop=True)


if __name__ == "__main__":
    main()
