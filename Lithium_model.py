from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



DATA_DIR = Path("data")

LCE_KG_PER_EV_BASE = 48.0
RECYCLING_SHARE_BASE = 0.05

USE_EV_SHARE_BASECASE = True
EV_SHARE_BASECASE = {
    2020: 0.62, 2021: 0.65, 2022: 0.67, 2023: 0.70, 2024: 0.73, 2025: 0.75,
    2026: 0.74, 2027: 0.73, 2028: 0.71, 2029: 0.70, 2030: 0.69,
}

START_YEAR = 2020
END_YEAR = 2030



def load_ev_sales_iea(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    s = df.set_index("year")["ev_sales_million"].sort_index()
    s.index = s.index.astype(int)
    return s


def load_supply_usgs(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    s = df.set_index("year")["supply_kt_lce"].sort_index()
    s.index = s.index.astype(int)
    return s


def years_range(start_year: int, end_year: int) -> pd.Index:
    return pd.Index(range(start_year, end_year + 1), name="year")


def extend_series_with_cagr(hist: pd.Series, years_index: pd.Index,
                            growth_rate: float, anchor_year: int) -> pd.Series:
    """
    過去データを指定した成長率で延長させる。
    """

    out = pd.Series(index=years_index, dtype=float)

    # 過去データをコピー
    for y, v in hist.items():
        if y in out.index:
            out.loc[y] = v

    anchor_val = out.loc[anchor_year]

    # 成長率で延長
    for y in years_index:
        if y > anchor_year:
            out.loc[y] = anchor_val * ((1 + growth_rate) ** (y - anchor_year))

    return out


def build_total_demand(ev_sales_million: pd.Series) -> pd.Series:
    """
    EV販売台数からリチウム総需要を計算する。
    ・EV用途以外も含めるため、EV比率から倍率を計算
    ・リサイクル分を差し引いた純需要を算出
    """

    ev_share = pd.Series(EV_SHARE_BASECASE).reindex(ev_sales_million.index)
    multiplier = 1 / ev_share

    ev_demand = ev_sales_million * LCE_KG_PER_EV_BASE
    total_demand = ev_demand * multiplier
    net = total_demand * (1 - RECYCLING_SHARE_BASE)

    net.name = "demand_kt_lce"
    return net


def run_scenario(years_index, ev_sales_hist, supply_hist,
                 ev_growth, supply_growth, anchor_year):
    """
    指定した成長率を用いて、
    需要と供給の推移を計算する。
    """

    ev_full = extend_series_with_cagr(
        ev_sales_hist, years_index,
        ev_growth, anchor_year
    )

    demand = build_total_demand(ev_full)

    supply = extend_series_with_cagr(
        supply_hist, years_index,
        supply_growth, anchor_year
    )
    supply.name = "supply_kt_lce"

    return pd.concat(
        [ev_full.rename("ev_sales_million"),
         demand,
         supply],
        axis=1
    )


# =========================
# Plot
# =========================
def plot_df(df: pd.DataFrame, title: str, save_path: Path | None = None):
    """
    需要と供給の推移をグラフに表示と保存
    """
    years = df.index.values

    plt.figure()
    plt.plot(years, df["demand_kt_lce"], label="Demand (kt LCE)")
    plt.plot(years, df["supply_kt_lce"], label="Supply (kt LCE)")
    plt.ylim(bottom=0)
    plt.xlabel("Year")
    plt.ylabel("kt LCE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()

def main():

    years_index = years_range(START_YEAR, END_YEAR)

    ev_sales_hist = load_ev_sales_iea(DATA_DIR / "ev_sales_iea.csv")
    supply_hist = load_supply_usgs(DATA_DIR / "supply_usgs.csv")

    # Use historical data through 2024; projections start from 2025
    anchor_year = 2024

    # IEA中期見通しに基づいて、EV成長率12%、供給成長率7%を固定する
    ev_growth = 0.12    # 12%
    supply_growth = 0.09 # 9%

    df = run_scenario(
        years_index,
        ev_sales_hist,
        supply_hist,
        ev_growth,
        supply_growth,
        anchor_year
    )

    plot_df(
        df,
        title=(
           "Lithium Supply–Demand Scenario Model\n"
    f"EV growth: {ev_growth*100:.1f}% | "
    f"Supply growth: {supply_growth*100:.1f}%"
        ),
        save_path=Path("images/lithium_supply_demand.png")
    )


if __name__ == "__main__":
    main()
    print("Saved to:", Path("images/lithium_supply_demand.png").resolve())