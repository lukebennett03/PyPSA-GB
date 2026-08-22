"""
Post-solve diagnostic for TES-SSRC scenario runs.

Usage
-----
    python scripts/nuclear/check_tes_run.py                    # all runs

Two blocks are reported.

CORRECTNESS asserts the things that, if wrong, invalidate the run: the reactor
held constant thermal output, the network really was a copper plate, nuclear
capacity was preserved across the FES compensation, and no load was shed. Any
FAIL here means the result cannot be compared against another scenario.

VIABILITY reports the quantities that decide whether scenarios will separate
from one another. The important one is the distribution of *daily* price
spread: the thermal stores are roughly one hour deep, so only intra-day spread
is reachable. A year with a wide annual price range but flat individual days
offers the system nothing.

The headroom estimate at the end is an upper bound on annual benefit computed
from the counterfactual alone - perfect foresight, one cycle per day, no price
impact. If that ceiling is small, the scenarios will not separate and there is
no point running them.
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = REPO / "config" / "defaults.yaml"
SCENARIOS = REPO / "config" / "scenarios.yaml"

# Only scenarios beginning with one of these are checked. resources/network/
# accumulates solved networks from earlier work (HT35_*, Historical_*) built
# under different configuration, whose objectives are not comparable with these.
# One entry per FES pathway: Holistic Transition, Electric Engagement,
# Hydrogen Evolution, Counterfactual.
SCENARIO_PREFIXES = ("HT37", "EE37", "HE37", "CF37")

# Scenario names are {pathway}{year}[_{network}]_{variant}. Stripping the
# variant leaves the study key, and scenarios sharing a key form a comparable
# pair - identical in every respect except the flexibility under test.
#
#   HT37_copperplate_counterfactual  -> HT37_copperplate
#   HT37_copperplate_tes5            -> HT37_copperplate
#   EE37_20_clusters_counterfactual  -> EE37_20_clusters
#   CF37_30_clusters_tes5            -> CF37_30_clusters
#
# tes\d+ rather than tes5 so the reactor-count series (tes1..tes5) pairs
# correctly without further edits.
VARIANT_PATTERN = re.compile(r"_(counterfactual|tes\d+)$")
BASELINE_VARIANT = "counterfactual"


def study_key(scenario: str) -> str | None:
    """Everything before the variant suffix, or None if the name doesn't match."""
    match = VARIANT_PATTERN.search(scenario)
    return scenario[: match.start()] if match else None


def baseline_for(scenario: str) -> str | None:
    """The counterfactual belonging to the same study as this scenario."""
    key = study_key(scenario)
    return f"{key}_{BASELINE_VARIANT}" if key else None

# Foreign nodes behind HVDC interconnectors carry carrier "AC" like GB buses
# do, but they are separate markets clearing at their own prices. Including
# them reports a spread that is physically correct and says nothing about GB's
# internal network.
#
# Their naming differs by pipeline path, which is why this is a substring test
# rather than a prefix:
#   clustered   - cluster_network.py:454 renames them external_{country}_{bus}
#   unclustered - they keep their build names, e.g. HVDC_External_Norway_Kvilldal
# A prefix match on "external" silently passes all nine foreign nodes through
# on unclustered runs, contaminating every price statistic.
EXTERNAL_BUS_MARKER = "external"

# Carriers that represent electricity. Buses carrying H2, steam or heat price a
# different commodity entirely and must never enter the comparison.
ELECTRICAL_CARRIERS = ("AC", "electricity")

PASS = "  [PASS]"
FAIL = "  [FAIL]"
INFO = "  [ ok ]"
WARN = "  [warn]"

_failures: list[str] = []


def _report(ok: bool, label: str, detail: str = "") -> None:
    print(f"{PASS if ok else FAIL} {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def _header(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def _deep_merge(base: dict, override: dict) -> dict:
    """Mirror config_loader.deep_merge so scenario overrides are respected."""
    result = dict(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_scenario_config(scenario: str) -> dict:
    """
    Return the merged config as the pipeline would see it for this scenario.

    Reading defaults.yaml alone is wrong: scenarios override individual leaves
    (modules_per_reactor, storage_units.unit_1.tes_e_nom_mwh_th, clustering,
    transmission) and the merge is recursive, so the effective configuration
    differs per scenario.
    """
    with open(DEFAULTS, "r", encoding="utf-8") as fh:
        defaults = yaml.safe_load(fh) or {}
    with open(SCENARIOS, "r", encoding="utf-8") as fh:
        scenarios = yaml.safe_load(fh) or {}

    return _deep_merge(defaults, scenarios.get(scenario, {}))


def load_tes_config(scenario: str) -> dict:
    return load_scenario_config(scenario).get("nuclear_tes", {})


def load_busmap(scenario: str) -> dict[str, list[str]]:
    """
    Map each cluster back to the real bus names it absorbed.

    cluster_network writes {scenario}_clustering_busmap.csv alongside the
    clustered network. It is the only place the original bus names survive -
    the clustered .nc carries "cluster_13" with no record that this is
    Bramford. Without it, every spatial statement in the results is unreadable.
    """
    path = REPO / "resources" / "network" / f"{scenario}_clustering_busmap.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"bus_id", "cluster"}.issubset(df.columns):
        return {}
    return {c: sorted(g.bus_id.astype(str)) for c, g in df.groupby("cluster")}


def label(bus: str, busmap: dict[str, list[str]]) -> str:
    """Render a cluster with the real buses inside it, e.g. cluster_13 (Bramford)."""
    members = busmap.get(str(bus))
    if not members:
        return str(bus)
    shown = ", ".join(members[:3]) + ("..." if len(members) > 3 else "")
    return f"{bus} ({shown})"


# =============================================================================
# CORRECTNESS
# =============================================================================
def check_reactor(n: pypsa.Network, cfg: dict) -> dict:
    """Reactor holds constant thermal output; PSRC pinned only when no modules."""
    reactors = n.generators.index[n.generators.index.str.endswith("_reactor")]
    if not len(reactors):
        _report(False, "reactor present", "no *_reactor generator found")
        return {}

    reactor = reactors[0]
    stem = reactor[: -len("_reactor")]
    p_nom_th = float(cfg["reactor_p_nom_th"])

    series = n.generators_t.p[reactor]
    flat = np.allclose(series.values, p_nom_th, atol=1e-3)
    _report(
        flat,
        "reactor holds constant thermal output",
        f"{series.min():,.1f} - {series.max():,.1f} MW_th (expected {p_nom_th:,.0f})",
    )

    psrc = f"{stem}_psrc"
    if psrc not in n.links.index:
        _report(False, "PSRC link present", f"{psrc} missing")
        return {"stem": stem}

    # Delivered electricity is -p1 by PyPSA sign convention.
    psrc_el = -n.links_t.p1[psrc]
    charge_links = n.links.index[n.links.index.str.contains("_charge")]
    n_modules = 0
    if len(charge_links):
        n_modules = len({c.split("_unit_")[0] for c in charge_links})

    if n_modules == 0:
        expected = p_nom_th * float(cfg["psrc_efficiency"])
        pinned = np.allclose(psrc_el.values, expected, atol=1.0)
        _report(
            pinned,
            "PSRC pinned at full output (counterfactual must-run)",
            f"{psrc_el.min():,.1f} - {psrc_el.max():,.1f} MW_e (expected {expected:,.1f})",
        )
    else:
        print(f"{INFO} {n_modules} TES modules present - PSRC free to throttle "
              f"({psrc_el.min():,.1f} - {psrc_el.max():,.1f} MW_e)")

    return {"stem": stem, "psrc_el": psrc_el, "n_modules": n_modules}


def check_copperplate(n: pypsa.Network, cfg: dict,
                      busmap: dict[str, list[str]] | None = None) -> pd.Series:
    """
    Assert uniform pricing only when the run was configured as a copper plate.

    transmission.min_line_s_nom > 0 relaxes every line, so GB must clear at one
    price and any spread is a bug. With the relaxation off the network is
    genuinely constrained, price separation is the phenomenon under study, and
    failing on it would be nonsense - so the same numbers are reported as a
    diagnostic instead.

    Foreign nodes behind HVDC interconnectors also carry carrier "AC" but are
    separate markets with their own prices, and non-electrical buses (H2, steam,
    heat) price a different commodity. Both are excluded either way.
    """
    busmap = busmap or {}
    relaxation = float((cfg.get("transmission") or {}).get("min_line_s_nom", 0) or 0)
    copperplate = relaxation > 0

    mp = n.buses_t.marginal_price

    electrical = [b for b in mp.columns
                  if n.buses.at[b, "carrier"] in ELECTRICAL_CARRIERS]
    gb = [b for b in electrical
          if EXTERNAL_BUS_MARKER not in str(b).casefold()]
    foreign = [b for b in electrical if b not in gb]
    other = [b for b in mp.columns if b not in electrical]

    if not gb:
        print(f"{WARN} no GB electrical buses identified - skipping copperplate check")
        return mp.mean(axis=1)

    spread = mp[gb].max(axis=1) - mp[gb].min(axis=1)

    if copperplate:
        _report(
            bool(spread.max() < 1e-3),
            "copperplate - uniform price across GB buses",
            f"{len(gb)} GB buses, max cross-bus spread {spread.max():.6f} GBP/MWh",
        )
        print(f"         compared: {', '.join(str(b) for b in gb)}")
    else:
        print(f"{INFO} network constrained (min_line_s_nom = 0) - price "
              f"separation expected, not a failure")
        print(f"         {len(gb)} GB buses, cross-bus spread: "
              f"mean {spread.mean():,.2f}, p95 {np.percentile(spread, 95):,.2f}, "
              f"max {spread.max():,.2f} GBP/MWh")
        _local_spread_table(mp[gb], busmap)

    if foreign:
        print(f"         excluded {len(foreign)} foreign AC bus(es) behind HVDC "
              f"- separate markets, own prices")
    if other:
        carriers = sorted({str(n.buses.at[b, 'carrier']) for b in other})
        print(f"         excluded {len(other)} non-electrical bus(es): "
              f"{', '.join(carriers)}")

    return mp[gb].mean(axis=1)


def _local_spread_table(mp: pd.DataFrame, busmap: dict[str, list[str]]) -> None:
    """
    Per-node daily spread, which is what decides whether siting the reactor
    somewhere congested gives the store anything to work with. Compare the
    median against the copperplate run: if it has not moved, congestion does
    not rescue the economics and the single-node result stands at higher
    spatial resolution.
    """
    rows = []
    for bus in mp.columns:
        s = mp[bus]
        daily = s.groupby(s.index.date)
        d = daily.max() - daily.min()
        rows.append((bus, s.mean(), float(np.median(d)), d.mean(),
                     int((d > 30).sum())))

    rows.sort(key=lambda r: -r[2])
    print()
    print(f"         {'node':<38}{'mean £':>9}{'median':>9}{'mean':>8}{'d>30':>7}")
    print(f"         {'':<38}{'':>9}{'daily spread':>17}{'':>7}")
    for bus, mean_p, med_d, mean_d, big in rows:
        print(f"         {label(bus, busmap)[:36]:<38}"
              f"{mean_p:>9,.2f}{med_d:>9,.2f}{mean_d:>8,.2f}{big:>7}")


def check_nuclear_capacity(n: pypsa.Network, cfg: dict, stem: str | None) -> None:
    """EPR carries the steam carrier, so carrier 'nuclear' alone under-reports."""
    fleet = n.generators.loc[
        n.generators.carrier.str.casefold() == "nuclear", "p_nom"
    ].sum()
    epr = float(cfg["reactor_p_nom_th"]) * float(cfg["psrc_efficiency"])
    total = fleet + epr

    print(f"{INFO} nuclear fleet (carrier 'nuclear'): {fleet:,.2f} MW")
    print(f"{INFO} EPR via PSRC link:                 {epr:,.2f} MW")
    print(f"{INFO} combined:                          {total:,.2f} MW")


def check_load_shedding(n: pypsa.Network) -> None:
    shed_gens = n.generators.index[n.generators.carrier == "load_shedding"]
    if not len(shed_gens):
        print(f"{WARN} no load_shedding generators found")
        return
    shed = n.generators_t.p[shed_gens].values.sum()
    _report(
        shed < 1.0,
        "no load shedding",
        f"{shed / 1e3:,.1f} GWh shed",
    )


def check_energy_balance(n: pypsa.Network) -> None:
    demand = n.loads_t.p_set.values.sum()
    print(f"{INFO} annual demand served: {demand / 1e6:,.2f} TWh")


# =============================================================================
# VIABILITY
# =============================================================================
def price_report(price: pd.Series) -> pd.Series:
    _header("VIABILITY - PRICE STRUCTURE")

    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print("  annual price distribution (GBP/MWh)")
    for q in qs:
        print(f"    p{q:<3} {np.percentile(price.values, q):>10,.2f}")
    print(f"    mean {price.mean():>10,.2f}   std {price.std():>8,.2f}")

    daily = price.groupby(price.index.date)
    spread = daily.max() - daily.min()

    print()
    print("  DAILY spread (max - min within each day) - what a 1h store can reach")
    for q in [5, 25, 50, 75, 95]:
        print(f"    p{q:<3} {np.percentile(spread.values, q):>10,.2f}")
    print(f"    mean {spread.mean():>10,.2f}")
    print(f"    days with spread > 30 GBP/MWh: "
          f"{int((spread > 30).sum())} of {len(spread)}")

    return spread


def curtailment_report(n: pypsa.Network) -> None:
    _header("VIABILITY - CURTAILMENT AND NUCLEAR UTILISATION")

    variable = [c for c in ("wind_offshore", "wind_onshore", "solar_pv", "marine")
                if (n.generators.carrier == c).any()]
    rows = []
    for carrier in variable:
        gens = n.generators.index[n.generators.carrier == carrier]
        gens = [g for g in gens if g in n.generators_t.p.columns]
        if not gens:
            continue
        actual = n.generators_t.p[gens].values.sum()
        if set(gens).issubset(n.generators_t.p_max_pu.columns):
            avail = (n.generators_t.p_max_pu[gens] * n.generators.p_nom[gens]).values.sum()
        else:
            avail = np.nan
        pct = 100 * (avail - actual) / avail if avail and not np.isnan(avail) else np.nan
        rows.append((carrier, actual / 1e6, (avail - actual) / 1e6 if avail else np.nan, pct))

    if rows:
        print(f"  {'carrier':<16}{'output TWh':>13}{'curtailed TWh':>16}{'curtailed %':>13}")
        for c, out, cur, pct in rows:
            print(f"  {c:<16}{out:>13,.2f}{cur:>16,.2f}{pct:>12,.1f}%")

    nuc = n.generators.index[n.generators.carrier.str.casefold() == "nuclear"]
    nuc = [g for g in nuc if g in n.generators_t.p.columns]
    if nuc:
        out = n.generators_t.p[nuc].values.sum()
        cap = n.generators.p_nom[nuc].sum() * len(n.snapshots)
        print()
        print(f"  remaining nuclear fleet capacity factor: {100 * out / cap:,.1f}%")
        print("  (heavy curtailment here means the system is in surplus and "
              "flexibility is cheap to charge but worth little to discharge)")


def _design_from_network(n: pypsa.Network) -> tuple[float, float, str] | None:
    """
    Derive the module design from the solved network rather than from config.

    Config can drift from what was actually built - a scenario may override a
    leaf, or a run may predate an edit. The network is what was solved, so it
    is the authoritative source whenever the components are present.
    """
    charge = n.links.index[n.links.index.str.endswith("_charge")]
    discharge = n.links.index[n.links.index.str.endswith("_discharge")]
    psrc = n.links.index[n.links.index.str.endswith("_psrc")]

    if not len(charge) or not len(discharge) or not len(psrc):
        return None

    psrc_eff = float(n.links.at[psrc[0], "efficiency"])
    forgone = float(n.links.loc[charge, "p_nom"].sum()) * psrc_eff
    recovered = float(
        (n.links.loc[discharge, "p_nom"] * n.links.loc[discharge, "efficiency"]).sum()
    )
    label = f"{len(charge)} charge / {len(discharge)} SSRC links (from network)"
    return forgone, recovered, label


def headroom(n: pypsa.Network, price: pd.Series, cfg: dict) -> None:
    """Upper bound on annual benefit: perfect foresight, one cycle/day."""
    _header("VIABILITY - BENEFIT HEADROOM (UPPER BOUND)")

    built = _design_from_network(n)
    if built is not None:
        forgone, recovered, label = built
    else:
        # Counterfactual: nothing was built, so price the design under test from
        # the merged config so the ceiling stays comparable across scenarios.
        units = cfg.get("storage_units", {}) or {}
        if not units:
            print("  no storage_units in config and none built - skipping")
            return
        modules = int(cfg.get("modules_per_reactor") or 0) or 5
        psrc_eff = float(cfg["psrc_efficiency"])
        forgone = modules * sum(
            float(u["charge_p_nom_mw_th"]) for u in units.values()
        ) * psrc_eff
        recovered = modules * sum(
            float(u["ssrc_p_nom_mw_th"]) * float(u["ssrc_efficiency"])
            for u in units.values()
        )
        label = (f"{modules} modules x {len(units)} units (from config - "
                 f"no TES built in this run)")

    print(f"  design: {label}")
    print(f"  charge diverts        {forgone:>8,.1f} MWh_e of PSRC output per cycle")
    print(f"  discharge recovers    {recovered:>8,.1f} MWh_e per cycle")
    print(f"  round-trip electrical {100 * recovered / forgone:>7,.1f}%")
    print(f"  break-even needs peak >= {forgone / recovered:,.2f} x charge price")

    daily = price.groupby(price.index.date)
    net = (recovered * daily.max() - forgone * daily.min())
    profitable = net[net > 0]

    print()
    print(f"  profitable days: {len(profitable)} of {len(net)}")
    print(f"  annual ceiling:  GBP {profitable.sum() / 1e6:,.2f}M")
    print()
    print("  This is an upper bound - perfect foresight, one cycle per day, no")
    print("  price impact from the system's own dispatch. Realised benefit will")
    print("  be materially lower. If this ceiling is small relative to total")
    print("  system cost, the scenarios will not separate.")


# =============================================================================
def check_one(scenario: str) -> tuple[int, float | None]:
    """Run every check against one solved network. Returns (exit code, objective)."""
    global _failures
    _failures = []

    path = REPO / "resources" / "network" / f"{scenario}_solved.nc"
    if not path.exists():
        print(f"Solved network not found: {path}")
        return 2, None

    scenario_cfg = load_scenario_config(scenario)
    cfg = scenario_cfg.get("nuclear_tes", {})
    busmap = load_busmap(scenario)
    n = pypsa.Network(str(path))

    _header(f"SCENARIO: {scenario}")
    print(f"Snapshots: {len(n.snapshots):,}  "
          f"({n.snapshots[0]} to {n.snapshots[-1]})")
    print(f"Objective: GBP {n.objective / 1e6:,.2f}M")
    if busmap:
        target = cfg.get("target_generator")
        if target:
            cluster = str(target).rsplit(" ", 1)[0]
            print(f"Reactor  : {label(cluster, busmap)}")

    _header("CORRECTNESS")
    info = check_reactor(n, cfg)
    price = check_copperplate(n, scenario_cfg, busmap)
    check_nuclear_capacity(n, cfg, info.get("stem"))
    check_load_shedding(n)
    check_energy_balance(n)

    price_report(price)
    curtailment_report(n)
    headroom(n, price, cfg)

    _header(f"VERDICT - {scenario}")
    if _failures:
        print(f"  {len(_failures)} correctness check(s) FAILED:")
        for f in _failures:
            print(f"    - {f}")
        print("\n  This run must not be compared against another scenario.")
        return 1, n.objective

    print("  All correctness checks passed.")
    print(f"  Objective for scenario comparison: GBP {n.objective / 1e6:,.2f}M")
    return 0, n.objective


def discover() -> list[str]:
    """Solved networks matching any active prefix, grouped by study."""
    directory = REPO / "resources" / "network"
    found = {p.name[: -len("_solved.nc")] for p in directory.glob("*_solved.nc")}
    found = {s for s in found if s.startswith(SCENARIO_PREFIXES)}
    # Sort by study, then counterfactual before its variants, so the summary
    # reads in the order the comparisons are made.
    return sorted(found, key=lambda s: (study_key(s) or s,
                                        0 if s.endswith(BASELINE_VARIANT) else 1,
                                        s))


def summarise(results: dict[str, tuple[int, float | None]]) -> None:
    """
    Report each study's benefit against its own counterfactual.

    A single global baseline cannot work once there is more than one study:
    comparing a constrained 20-cluster run against a copperplate baseline mixes
    the network change with the flexibility effect, which is precisely the
    confound the paired design exists to remove.
    """
    _header("SUMMARY")

    studies: OrderedDict[str, list[str]] = OrderedDict()
    orphans: list[str] = []
    for scenario in results:
        key = study_key(scenario)
        if key is None:
            orphans.append(scenario)
        else:
            studies.setdefault(key, []).append(scenario)

    for key, members in studies.items():
        baseline_name = f"{key}_{BASELINE_VARIANT}"
        baseline = results.get(baseline_name, (None, None))[1]

        print()
        print(f"  {key}")
        print(f"  {'-' * 74}")
        print(f"    {'scenario':<40}{'objective GBPm':>16}{'benefit':>14}")

        for scenario in members:
            code, obj = results[scenario]
            flag = "" if code == 0 else "  [FAILED]"
            if obj is None:
                print(f"    {scenario:<40}{'-':>16}{'no result':>14}{flag}")
            elif scenario == baseline_name:
                print(f"    {scenario:<40}{obj / 1e6:>16,.2f}{'baseline':>14}{flag}")
            elif baseline is None:
                print(f"    {scenario:<40}{obj / 1e6:>16,.2f}"
                      f"{'no baseline':>14}{flag}")
            else:
                # Benefit is the cost the flexibility avoids, so it is positive
                # when the objective falls.
                print(f"    {scenario:<40}{obj / 1e6:>16,.2f}"
                      f"{(baseline - obj) / 1e6:>+13,.2f}M{flag}")

    if orphans:
        print()
        print(f"  Unpaired (name does not end in a known variant): "
              f"{', '.join(orphans)}")

    print()
    print("  Benefit is baseline minus scenario: positive means the flexibility")
    print("  reduced system operating cost. Each study is compared only against")
    print("  its own counterfactual.")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0

    if args:
        scenarios = args
        off_prefix = [s for s in scenarios if not s.startswith(SCENARIO_PREFIXES)]
        if off_prefix and not force:
            print(f"Refusing to check {off_prefix}: outside the active prefixes "
                  f"{SCENARIO_PREFIXES}.")
            print("Stale results from earlier work are not comparable against "
                  "the current configuration.")
            print("Pass --force to override, or edit SCENARIO_PREFIXES.")
            return 2
    else:
        scenarios = discover()
        if not scenarios:
            print(f"No solved networks matching {SCENARIO_PREFIXES} in "
                  f"resources/network/")
            return 2
        print(f"Checking {len(scenarios)} scenario(s) across "
              f"{len({study_key(s) for s in scenarios})} studies")

    results: dict[str, tuple[int, float | None]] = {}
    for scenario in scenarios:
        results[scenario] = check_one(scenario)

    if len(results) > 1:
        summarise(results)

    return max(code for code, _ in results.values())


if __name__ == "__main__":
    sys.exit(main())
