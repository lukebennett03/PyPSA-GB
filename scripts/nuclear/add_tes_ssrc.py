"""
Add TES-SSRC modules to the network that couple with nuclear reactors.
"""

import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
import logging
import warnings
import yaml
from typing import Optional

from pandas import DataFrame

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

# Suppress PyPSA warnings
warnings.filterwarnings('ignore', message='The network has not been optimized yet')

# Import logging configuration
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("add_tes_ssrc")
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

_CARRIER_SPEC = {
    "steam":   ("nuclear_steam",      "Nuclear Steam",      "#C71585"),
    "storage": ("nuclear_tes",        "Nuclear TES",        "#13B7ED"),
    "psrc":    ("nuclear_psrc",       "Nuclear PSRC",       "#8B008B"),
    "charge":  ("nuclear_tes_charge", "Nuclear TES Charge", "#5FAF5F"),
    "ssrc":    ("nuclear_ssrc",       "Nuclear SSRC",       "#E8843C"),
}


# =============================================================================
# EXTRACT PARAMETERS FROM DEFAULTS
# =============================================================================
DEFAULTS_PATH = Path(__file__).resolve().parents[2] / "config" / "defaults.yaml"


def load_tes_config(config: Optional[dict] = None) -> dict:
    """
    Return the ``nuclear_tes`` block from the merged config.

    Under Snakemake the merged config is passed in from the rule. Standalone,
    config/defaults.yaml is read directly so the script can be run on its own.

    Args:
        config: Merged config dict, or None to read defaults.yaml

    Returns:
        The nuclear_tes block (empty dict if absent)
    """
    if config is None:
        with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    return config.get("nuclear_tes", {})


# =============================================================================
# HELPERS
# =============================================================================
def _resolve_target(
        network: pypsa.Network,
        target_generator: str
) -> dict:
    """
    This functions takes in the target generator and returns necessary info
    Args:
        network: pypsa.Network
        target_generator: generator name

    Returns:
        dict containing name, electrical bus with (x,y) coordinates and p_nom
    """
    target_dict = {
        "name": None,
        "electrical_bus": None,
        "bus_x": None,
        "bus_y": None,
        "p_nom": None,
    }

    carriers = (
        network.generators["carrier"]
        .fillna("")
        .astype(str)
        .str.casefold()
    )

    nuclear = carriers.eq("nuclear")
    nuclear_gens = network.generators[nuclear]

    if target_generator in nuclear_gens.index:
        row = nuclear_gens.loc[target_generator]
        target_dict["name"] = target_generator
        bus = row["bus"]
        target_dict["electrical_bus"] = bus
        target_dict["bus_x"] = network.buses.loc[bus, "x"]
        target_dict["bus_y"] = network.buses.loc[bus, "y"]
        target_dict["p_nom"] = row["p_nom"]
    else:
        raise ValueError("Invalid target generator")

    return target_dict

def _cell_names(
        gen: str,
        module: int,
        unit: str,
) -> dict:
    """
    Component names for a single (module, unit) cell of the TES-SSRC grid.

    Every name for one storage train is derived here, so the bus created by
    the bus builder and the bus referenced by the store and links cannot
    diverge. Callers must not construct these names themselves.

    Args:
        gen: Generator name the module belongs to
        module: 1-indexed module number
        unit: Storage unit key from the config (e.g. "unit_1")

    Returns:
        dict with keys: bus, store, charge, discharge
    """
    stem = f"{gen}_{module}_{unit}"

    return {
        "bus": f"{stem}_bus",
        "store": f"{stem}_store",
        "charge": f"{stem}_charge",
        "discharge": f"{stem}_discharge",
    }


def _names(
        gen: str,
        modules: int,
        units: list,
) -> dict:
    """
    Component names across the whole TES-SSRC grid.

    A thin wrapper over :func:`_cell_names`, used for validation counts and
    uniqueness checks. The builder itself should call ``_cell_names`` per
    cell rather than pairing these lists by index.

    Args:
        gen: Generator name
        modules: Number of modules per reactor (0 gives the counterfactual)
        units: Storage unit keys from the config

    Returns:
        dict of lists with keys: buses, stores, charges, discharges
    """
    names = {
        "buses": [],
        "stores": [],
        "charges": [],
        "discharges": [],
    }

    for module in range(1, modules + 1):
        for unit in units:
            cell = _cell_names(gen, module, unit)
            names["buses"].append(cell["bus"])
            names["stores"].append(cell["store"])
            names["charges"].append(cell["charge"])
            names["discharges"].append(cell["discharge"])

    return names

def _add_carrier(
        network: pypsa.Network,
        cfg: dict
) -> dict:
    """
    Declare the TES-SSRC carriers and return the resolved names

    Args:
        network: pypsa.Network
        cfg: The full nuclear_tes block

    Returns
        Mapping of role to resolved carrier name, e.g. {"steam": "nuclear_steam"}
    """
    configured = cfg.get("carriers", {}) or {}
    resolved = {}

    for role, (default_name, nice_name, colour) in _CARRIER_SPEC.items():
        name = configured.get(role, default_name)
        resolved[role] = name

        if name in network.carriers.index:
            logger.debug(f"Carrier {name} already exists. Leaving unchanged.")
            continue

        network.add(
            "Carrier", name,
            nice_name=nice_name,
            color=colour,
            co2_emissions=0.0,
        )
        logger.info(f"Added carrier {name} ({nice_name})")

    return resolved

def _add_busses(
        network: pypsa.Network,
        target_dict: dict,
        carriers: dict,
        modules: int,
        units: list,
) -> str:
    """
    Add the steam header bus and one thermal store bus per module-unit cell.

    All new buses inherit the electrical bus coordinates, since the storage
    equipment sits at the same site as the reactor. Coordinates do not enter
    the optimisation, but spatial clustering uses them, so sharing them keeps
    the stores in the same cluster as their parent bus.

    TES bus names come from :func:`_cell_names` so they cannot diverge from
    the stores and links attached to them.

    Args:
        network: PyPSA network, modified in place
        target_dict: Spec from :func:`_resolve_target`, providing the
            generator name and the electrical bus coordinates
        carriers: Role-to-name mapping from :func:`_add_carriers`
        modules: Modules per reactor; 0 adds the steam bus only
        units: Storage unit keys from the config

    Returns:
        Name of the steam header bus, needed by the PSRC and charge links
    """
    target_generator = target_dict["name"]

    # The steam and tes buses will use the electricity coordinates
    # to represent that the modules are at the same locations
    bus_x = target_dict["bus_x"]
    bus_y = target_dict["bus_y"]

    steam_carrier = carriers["steam"]
    store_carrier = carriers["storage"]

    steam_bus = f"{target_generator}_steam"

    if steam_bus not in network.buses.index:
        network.add("Bus", steam_bus,
                    carrier=steam_carrier,
                    x=bus_x, y=bus_y,
                    v_nom=1.0)
        logger.info(f"Added {steam_bus} bus")

    tes_count = 0
    for module in range(1, modules + 1):
        for unit in units:
            cell = _cell_names(target_generator, module, unit)
            tes_bus = cell["bus"]

            if tes_bus not in network.buses.index:
                network.add("Bus", tes_bus,
                            carrier=store_carrier,
                            x=bus_x, y=bus_y,
                            v_nom=1.0)
                tes_count += 1

    logger.info(f"{target_generator}: added {tes_count} TES busses")

    return steam_bus

def _handle_fes_nuclear(
        network: pypsa.Network,
        target_dict: dict,
        cfg: dict
) -> dict:
    """
    Remove the target FES generator and rescale the remaining nuclear fleet.

    Two changes are needed before the reactor can be added:

    1. The target generator is removed, so the plant's electricity reaches the
       grid only through the PSRC link rather than directly.
    2. The remaining nuclear generators are scaled so total system nuclear
       capacity is unchanged. Without this, replacing a ~250 MW FES allocation
       with a 1610 MW EPR raises GB nuclear by over 1 GW, which alters the
       residual demand pattern and inflates the apparent value of flexibility.

    The scale factor follows from requiring the post-change total to equal the
    original::

        f = (T - E) / (T - t)

    where T is total nuclear capacity, t the target generator's capacity and E
    the EPR's electrical output. E is derived from the thermal rating and PSRC
    efficiency rather than read from config, so it cannot drift from the link
    actually built.

    Must run before :func:`_add_reactor_psrc`.

    Args:
        network: PyPSA network, modified in place
        target_dict: Spec from :func:`_resolve_target`
        cfg: The full nuclear_tes block

    Returns:
        Summary of the change: total capacity before and after, the scale
        factor applied, the target's capacity and the EPR's electrical output

    Raises:
        ValueError: If the EPR exceeds total nuclear capacity, so no positive
            scale factor exists; or if the target is the only nuclear unit
    """
    target_generator = target_dict["name"]

    # Get all other generators
    carriers = (
        network.generators["carrier"]
        .fillna("")
        .astype(str)
        .str.casefold()
    )

    nuclear = carriers.eq("nuclear")
    nuclear_gens = network.generators[nuclear]

    # Extract target
    target = nuclear_gens[nuclear_gens.index == target_generator]

    # -------------------------------------------------------------------------
    # Rescale all generators
    # -------------------------------------------------------------------------
    total_capacity = nuclear_gens.p_nom.sum()
    target_capacity = nuclear_gens.at[target_generator, "p_nom"]
    epr_elec_capacity = cfg["reactor_p_nom_th"] * cfg["psrc_efficiency"]

    # Guards
    if total_capacity < epr_elec_capacity:
        raise ValueError(f"Total model capacity: {total_capacity} must be "
                         f"greater than the scenario capacity: {epr_elec_capacity}")
    if total_capacity == target_capacity:
        raise ValueError(f"Total model capacity: {total_capacity} cannot be equal to"
                         f"the target model capacity: {target_capacity}")

    scale_factor = (total_capacity - epr_elec_capacity) / (total_capacity - target_capacity)

    # -------------------------------------------------------------------------
    # Remove target generator
    # -------------------------------------------------------------------------
    network.remove("Generator", target.index.tolist())
    logger.info(f"Removed {target_generator} generator")

    # -------------------------------------------------------------------------
    # Scale remaining
    # -------------------------------------------------------------------------
    remaining = [g for g in nuclear_gens.index if g != target_generator]
    network.generators.loc[remaining, "p_nom"] *= scale_factor

    new_total = network.generators.loc[remaining, "p_nom"].sum() + epr_elec_capacity

    # Check scaling has worked
    if not np.isclose(new_total, total_capacity, rtol=1e-9, atol=1e-6):
        raise ValueError(f"After scaling total should be equal. Currently total before scaling: {total_capacity} "
                         f" and total after scaling: {new_total}")

    return {
        "total_before": total_capacity,
        "total_after": new_total,
        "scale_factor": scale_factor,
        "target_capacity": target_capacity,
        "epr_capacity_el": epr_elec_capacity,
    }


def _add_reactor_psrc(
        network: pypsa.Network,
        steam_bus: str,
        resolved_carriers: dict,
        cfg: dict,
        target_dict: dict,
) -> None:
    """
    Add the reactor and its primary steam Rankine cycle.

    The reactor is a Generator on the steam bus producing constant thermal
    power - ``p_min_pu`` and ``p_max_pu`` are both 1.0, which is what makes
    this a storage-coupled architecture rather than direct reactor flexing.
    Electricity reaches the grid through the PSRC link, whose ``p_nom`` is the
    thermal rating at bus0; output is ``p_nom x efficiency``.

    The reactor carries the steam carrier rather than ``nuclear``, so the
    fleet rescaling in :func:`_handle_fes_nuclear` cannot catch it.

    Args:
        network: PyPSA network, modified in place
        steam_bus: Steam header bus from :func:`_add_busses`
        resolved_carriers: Role-to-name mapping from :func:`_add_carriers`
        cfg: The full nuclear_tes block
        target_dict: Spec from :func:`_resolve_target`

    Raises:
        RuntimeError: If the original FES generator is still present, meaning
            :func:`_handle_fes_nuclear` has not run
    """
    target_generator = target_dict["name"]

    # Cannot add psrc unless original generator been removed
    if target_generator in network.generators.index:
        raise RuntimeError(
            f"{target_generator} still present - _handle_fes_nuclear must run first"
        )

    # Extract electricity bus
    elec_bus = target_dict["electrical_bus"]

    # Extract PSRC config parameters
    psrc_eff = cfg["psrc_efficiency"]
    reactor_p_nom_th = cfg["reactor_p_nom_th"]
    psrc_p_min_pu = cfg["psrc_p_min_pu"]

    network.add("Link", f"{target_generator}_psrc",
                bus0=steam_bus,
                bus1=elec_bus,
                carrier=resolved_carriers["psrc"],
                efficiency=psrc_eff,
                p_nom=reactor_p_nom_th,
                p_min_pu=psrc_p_min_pu,
                )
    logger.info(f"Added PSRC link")

    # Add generator
    network.add("Generator", f"{target_generator}_reactor",
                bus=steam_bus,
                carrier=resolved_carriers["steam"],
                p_nom=reactor_p_nom_th,
                p_min_pu=1.0,
                p_max_pu=1.0,
                )
    logger.info(f"Added reactor generator")

def _add_module(
        network: pypsa.Network,
        target_dict: dict,
        module: int,
        storage_units: dict,
        steam_bus: str,
        resolved_carriers: dict,

) -> None:
    """
    Add one TES-SSRC module: a charge link, store and SSRC link per unit.

    A module is the pair of storage trains described by Al Kindi et al. Each
    unit gets three components:

    1. A charge link diverting steam from the header to its thermal store
    2. The store itself, holding heat between charge and discharge
    3. An SSRC link converting stored heat back to electricity

    Link ``p_nom`` values are thermal and measured at bus0; electrical output
    is ``p_nom x efficiency``. The two units differ only in SSRC efficiency
    (0.296 and 0.237), reflecting the different steam grades they are charged
    from - a difference this model carries in the conversion efficiency rather
    than in separate steam states.

    Stores are cyclic, so state of charge returns to its starting value. Without
    this the optimiser could drain the stores over the horizon and treat the
    stored energy as free, flattering the flexible scenario.

    All component names come from :func:`_cell_names`, so the bus created by
    :func:`_add_busses` and the components attached to it cannot diverge.

    Args:
        network: PyPSA network, modified in place
        target_dict: Spec from :func:`_resolve_target`
        module: 1-indexed module number
        storage_units: Mapping of unit key to its parameters, from config
        steam_bus: Steam header bus from :func:`_add_busses`
        resolved_carriers: Role-to-name mapping from :func:`_add_carriers`
    """
    target_generator = target_dict["name"]
    elec_bus = target_dict["electrical_bus"]

    for unit, params in storage_units.items():
        cell = _cell_names(target_generator, module, unit)
        tes_bus = cell["bus"]

        # Build TES charging link
        network.add("Link", cell["charge"],
                    bus0=steam_bus,
                    bus1=tes_bus,
                    carrier=resolved_carriers["charge"],
                    efficiency=params["charge_efficiency"],
                    p_nom=params["charge_p_nom_mw_th"],)

        # Build TES storage
        network.add("Store", cell["store"],
                    bus=tes_bus,
                    carrier=resolved_carriers["storage"],
                    e_nom=params["tes_e_nom_mwh_th"],
                    e_cyclic=True)

        # Build SSRC link (discharge)
        network.add("Link", cell["discharge"],
                    bus0=tes_bus,
                    bus1=elec_bus,
                    carrier=resolved_carriers["ssrc"],
                    efficiency=params["ssrc_efficiency"],
                    p_nom=params["ssrc_p_nom_mw_th"],)


    logger.info(f"Added {len(storage_units)} TES charging links, TES storage and SSRC links")

def _validate(
        network: pypsa.Network,
        target_dict: dict,
        carriers: dict,
        modules: int,
        storage_units: dict,
        cfg: dict,
        nuclear_summary: dict,
        names: dict
) -> dict:
    """
    This function will be used to validate final modular setup. Staged for development

    Args:
        network:

    Returns:

    """
    target_reactor = target_dict["name"]

    # -------------------------------------------------------------------------
    # Check all new components exist
    # -------------------------------------------------------------------------
    # _names knows every component the build should have produced, so checking
    # that set against the network validates the names as well as the count.
    # A component created with a hand-built name rather than via _cell_names
    # shows up here as missing.
    #
    # Charge and discharge links share network.links - PyPSA keeps one frame
    # per component type, and both are Links.
    frames = {
        "buses": network.buses.index,
        "stores": network.stores.index,
        "charges": network.links.index,
        "discharges": network.links.index,
    }

    missing = {}

    for key, expected_names in names.items():
        present = frames[key]
        absent = [name for name in expected_names if name not in present]

        if absent:
            missing[key] = {
                "expected": len(expected_names),
                "found": len(expected_names) - len(absent),
                "absent": absent,
            }

    if missing:
        detail = "; ".join(
            f"{key}: {info['found']} of {info['expected']} present, "
            f"missing {info['absent']}"
            for key, info in missing.items()
        )
        raise ValueError(
            f"{target_reactor}: expected components absent from network - {detail}"
        )

    logger.info(
        f"{target_reactor}: all "
        f"{sum(len(v) for v in names.values())} expected components present"
    )

    # -------------------------------------------------------------------------
    # No unexpected extras
    # -------------------------------------------------------------------------
    # The existence check above catches missing components; this catches extra
    # ones - duplicates from a re-run, or spurious additions. Together they pin
    # the set exactly.
    expected_per_carrier = {
        carriers["charge"]: modules * len(storage_units),
        carriers["ssrc"]: modules * len(storage_units),
        carriers["psrc"]: 1,
    }

    for carrier, expected in expected_per_carrier.items():
        found = (network.links.carrier == carrier).sum()
        if found != expected:
            raise ValueError(
                f"{target_reactor}: expected {expected} links with carrier "
                f"{carrier}, found {found}"
            )

    n_stores = (network.stores.carrier == carriers["storage"]).sum()
    if n_stores != modules * len(storage_units):
        raise ValueError(
            f"{target_reactor}: expected {modules * len(storage_units)} stores, "
            f"found {n_stores}"
        )

    # -------------------------------------------------------------------------
    # Referential integrity
    # -------------------------------------------------------------------------
    # PyPSA creates links to nonexistent buses without complaint. The failure
    # surfaces only at solve time, with an error that does not name the cause.
    dangling = []

    for store, bus in network.stores["bus"].items():
        if bus not in network.buses.index:
            dangling.append(f"Store {store} -> {bus}")

    for column in ("bus0", "bus1"):
        for link, bus in network.links[column].items():
            if bus not in network.buses.index:
                dangling.append(f"Link {link}.{column} -> {bus}")

    if dangling:
        raise ValueError(
            f"{target_reactor}: components reference missing buses - {dangling}"
        )

    # -------------------------------------------------------------------------
    # Reactor holds constant thermal output
    # -------------------------------------------------------------------------
    # p_min_pu == p_max_pu == 1.0 is what makes this storage-coupled rather than
    # direct reactor flexing. Without it the optimiser throttles the reactor
    reactor = f"{target_reactor}_reactor"

    if reactor not in network.generators.index:
        raise ValueError(f"{reactor} not found in network")

    reactor_row = network.generators.loc[reactor]

    if not (reactor_row["p_min_pu"] == 1.0 and reactor_row["p_max_pu"] == 1.0):
        raise ValueError(
            f"{reactor}: expected p_min_pu = p_max_pu = 1.0, found "
            f"{reactor_row['p_min_pu']} and {reactor_row['p_max_pu']}"
        )

    if reactor_row["p_nom"] != cfg["reactor_p_nom_th"]:
        raise ValueError(
            f"{reactor}: expected p_nom {cfg['reactor_p_nom_th']} MW_th, "
            f"found {reactor_row['p_nom']}"
        )

    # The rescaling in _handle_fes_nuclear filters on carrier "nuclear". If the
    # reactor carried it, a re-run would shrink the EPR along with the fleet.
    if reactor_row["carrier"] == "nuclear":
        raise ValueError(
            f"{reactor}: carrier must differ from 'nuclear' or fleet rescaling "
            f"will catch it"
        )

    # -------------------------------------------------------------------------
    # Capacity arithmetic
    # -------------------------------------------------------------------------
    # Link p_nom is thermal and measured at bus0; electrical output is
    # p_nom x efficiency. Catches the p_nom / p_nom_th class of error, where a
    # link silently ends up with zero capacity and the model still solves.
    psrc = network.links.loc[f"{target_reactor}_psrc"]
    psrc_el = psrc["p_nom"] * psrc["efficiency"]
    expected_psrc_el = cfg["reactor_p_nom_th"] * cfg["psrc_efficiency"]

    if abs(psrc_el - expected_psrc_el) > 1.0:
        raise ValueError(
            f"PSRC delivers {psrc_el:.1f} MW_e, expected {expected_psrc_el:.1f}"
        )

    ssrc_links = network.links[network.links.carrier == carriers["ssrc"]]
    ssrc_el = (ssrc_links["p_nom"] * ssrc_links["efficiency"]).sum()
    expected_ssrc_el = modules * sum(
        u["ssrc_p_nom_mw_th"] * u["ssrc_efficiency"] for u in storage_units.values()
    )

    if abs(ssrc_el - expected_ssrc_el) > 1.0:
        raise ValueError(
            f"SSRC delivers {ssrc_el:.1f} MW_e, expected {expected_ssrc_el:.1f}"
        )

    # -------------------------------------------------------------------------
    # Coordinates
    # -------------------------------------------------------------------------
    # NaN coordinates break spatial plotting and can misplace buses during
    # clustering, separating the stores from the reactor they belong to.
    new_buses = network.buses[
        network.buses.carrier.isin([carriers["steam"], carriers["storage"]])
    ]

    no_coords = new_buses[new_buses[["x", "y"]].isna().any(axis=1)]

    if not no_coords.empty:
        raise ValueError(
            f"{target_reactor}: buses without coordinates - "
            f"{no_coords.index.tolist()}"
        )

    # -------------------------------------------------------------------------
    # Nuclear capacity preserved
    # -------------------------------------------------------------------------
    # The sole purpose of the rescaling in _handle_fes_nuclear.
    before = nuclear_summary["total_before"]
    after = nuclear_summary["total_after"]

    if abs(after - before) > 1.0:
        raise ValueError(
            f"Nuclear capacity changed: {before:.1f} MW before, "
            f"{after:.1f} MW after (scale factor "
            f"{nuclear_summary['scale_factor']:.4f})"
        )

    # -------------------------------------------------------------------------
    # PyPSA structural check
    # -------------------------------------------------------------------------
    network.consistency_check()

    summary = {
        "components_checked": sum(len(v) for v in names.values()),
        "psrc_mw_el": psrc_el,
        "ssrc_mw_el": ssrc_el,
        "plant_max_mw_el": psrc_el + ssrc_el,
        "nuclear_total_mw": after,
    }

    return summary

def _build_modules(
        network: pypsa.Network,
        target_generator: str,
        modules: int,
        storage_units: dict,
        cfg
) -> pypsa.Network:
    """
    Build a network containing modular setup.

    Args:
        network: PyPSA network
        target_generator: Target generator for flex
        modules: Number of modules to use
        storage_units: Mapping of unit key to its parameters, from config
        cfg: Configuration file

    Returns:
        Modified network
    """
    # -------------------------------------------------------------------------
    # Extract key info about target generator
    # -------------------------------------------------------------------------
    target_generator_dict = _resolve_target(
        network,
        target_generator=target_generator
    )

    # -------------------------------------------------------------------------
    # Extract module names
    # -------------------------------------------------------------------------
    names = _names(
        gen=target_generator,
        modules=modules,
        units=storage_units,
    )
    logger.info(f"Names of built components: {names}")

    # -------------------------------------------------------------------------
    # Build and extract carriers
    # -------------------------------------------------------------------------
    carriers = _add_carrier(
        network=network,
        cfg=cfg,
    )

    # -------------------------------------------------------------------------
    # Build all busses and extract the steam bus
    # -------------------------------------------------------------------------
    steam_bus = _add_busses(
        network=network,
        target_dict=target_generator_dict,
        carriers=carriers,
        modules=modules,
        units=storage_units,
    )

    # -------------------------------------------------------------------------
    # Re-configure nuclear reactors
    # -------------------------------------------------------------------------
    new_nuclear_summary = _handle_fes_nuclear(
        network=network,
        target_dict=target_generator_dict,
        cfg=cfg,
    )
    logger.info(f"New nuclear summary: {new_nuclear_summary}")

    # -------------------------------------------------------------------------
    # Build reactor and PSRC
    # -------------------------------------------------------------------------
    _add_reactor_psrc(
        network=network,
        steam_bus=steam_bus,
        resolved_carriers=carriers,
        cfg=cfg,
        target_dict=target_generator_dict,
    )

    # -------------------------------------------------------------------------
    # Build the modules
    # -------------------------------------------------------------------------
    for module in range(1, modules + 1):
        _add_module(
            network=network,
            target_dict=target_generator_dict,
            module=module,
            storage_units=storage_units,
            steam_bus=steam_bus,
            resolved_carriers=carriers
        )

    # -------------------------------------------------------------------------
    # Validate results
    # -------------------------------------------------------------------------
    validation_info = _validate(
        network=network,
        target_dict=target_generator_dict,
        carriers=carriers,
        modules=modules,
        storage_units=storage_units,
        cfg=cfg,
        nuclear_summary=new_nuclear_summary,
        names=names,
    )

    logger.info(f"{target_generator}: validation passed - {validation_info}")
    return network

# =============================================================================
# SCRIPT HELPERS
# =============================================================================
def _inspect(
        network: pypsa.Network,
        cfg: dict
) -> None:
    """
    Check new network setup containing TEC-SSRC modules.

    Args:
        network: PyPSA network
        cfg: Configuration file
    """
    # Filter to new carrier
    carriers = _add_carrier(network, cfg)
    new = set(carriers.values())

    # Inspect the constructed network
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # New components filtered by new carriers
    buses = network.buses[network.buses.carrier.isin(new)]
    links = network.links[network.links.carrier.isin(new)]
    stores = network.stores[network.stores.carrier.isin(new)]
    gens = network.generators[network.generators.carrier.isin(new)]

    # Inspect
    logger.info("\n--- BUSES ---")
    logger.info(buses[["carrier", "x", "y", "v_nom"]])

    logger.info("\n--- LINKS ---")
    logger.info(
        links[["bus0", "bus1", "carrier", "p_nom", "efficiency"]]
        .assign(mw_el=lambda d: d.p_nom * d.efficiency)
    )

    logger.info("\n--- STORES ---")
    logger.info(stores[["bus", "carrier", "e_nom", "e_cyclic"]])

    logger.info("\n--- REACTOR ---")
    logger.info(gens[["bus", "carrier", "p_nom", "p_min_pu", "p_max_pu"]])

    # Check nuclear capacity from original carriers. Should reveal a gap
    # that is filled by flexible generator to make up total capacity
    nuclear = network.generators[network.generators.carrier.str.casefold() == "nuclear"]
    logger.info(f"\nFleet: {len(nuclear)} generators, {nuclear.p_nom.sum():.1f} MW")


def _solve_and_report(
        network: pypsa.Network,
        cfg: dict
) -> None:
    """
    Solve network: Goal is to verify thermal output remains constant while
    electrical output is flexible.

    Args:
        network: PyPSA network
        cfg: Configuration file
    """
    status, condition = network.optimize(solver_name="highs")
    logger.info(f"{status}, {condition}")

    if status == "ok" and condition == "optimal":
        n = network
        carriers = _add_carrier(network, cfg)
        logger.info("\n--- REACTOR ---")
        logger.info(n.generators_t.p["FES_nuclear_NGET_Melksham_reactor"].describe())
        logger.info(n.stores_t.e.describe())

        psrc = "FES_nuclear_NGET_Melksham_psrc"
        ssrc = n.links[n.links.carrier == carriers["ssrc"]].index

        logger.info("\n--- STORES --- ")
        # p1 is negative at the output bus, so negate for generation
        elec = -(n.links_t.p1[psrc] + n.links_t.p1[ssrc].sum(axis=1))
        logger.info(elec.describe())

# =============================================================================
# BUILD FULL MODULAR SETUP
# =============================================================================
def build_tes_ssrc(
        network: pypsa.Network,
        tes_config: dict,
) -> pypsa.Network:
    """
    Build the TES-SSRC modules to couple to nuclear reactors.

    Args:
        network: PyPSA network
        tes_config: The nuclear_tes block from defaults.yaml

    Returns:
        Modified network with one steam bus and 5 x modules_per_reactor
        thermal store buses per site
    """
    logger.info("=" * 60)
    logger.info("ADDING TES-SSRC MODULES")
    logger.info("=" * 60)

    if not tes_config.get("enabled", False):
        logger.info("nuclear_tes.enabled is false - skipping TES-SSRC system")
        return network

    flex_generator = tes_config.get("target_generator", [])
    if not flex_generator:
        logger.warning("nuclear_tes.target_generator is empty - skipping TES-SSRC system")
        return network

    modules_per_reactor = tes_config["modules_per_reactor"]
    storage_units = tes_config["storage_units"]

    logger.info(f"Building {modules_per_reactor} modules of "
                f"{len(storage_units)} units at {flex_generator}")

    # Build network with modules added
    modular_network = _build_modules(
        network=network,
        target_generator=flex_generator,
        modules=modules_per_reactor,
        storage_units=storage_units,
        cfg=tes_config,
    )

    return modular_network

def main():
    """Main entry point when run as script."""

    # Check if running under snakemake
    if 'snakemake' in globals():
        # Get inputs/outputs from Snakemake
        network_file = snakemake.input.network
        output_file = snakemake.output.network
        tes_config = snakemake.params.nuclear_tes
        smoke_test = False

        # Get parameters
        scenario = snakemake.params.scenario

        logger.info(f"Adding TES-SSRC modular system for scenario: {scenario}")

    else:
        # Standalone testing
        import argparse
        parser = argparse.ArgumentParser(description="Add TES-SSRC modules to PyPSA network")
        parser.add_argument("--network", required=True, help="Input network file")
        parser.add_argument("--output", required=True, help="Output network file")
        parser.add_argument("--target_generator", required=True, help="Target generator")
        args = parser.parse_args()

        network_file, output_file = args.network, args.output
        tes_config = {**load_tes_config(), "enabled": True, "target_generator": args.target_generator}
        smoke_test = True

    # Load network
    logger.info(f"Loading network from: {network_file}")
    network = load_network(network_file)

    if smoke_test:
        # Set solve period to first week of Jan
        network.set_snapshots(network.snapshots[:168])

    # Construct TES-SSRC modules on network
    network = build_tes_ssrc(
        network=network,
        tes_config=tes_config,
    )

    if smoke_test:
        _inspect(network=network, cfg=tes_config)
        _solve_and_report(network=network, cfg=tes_config)

    # Save network
    logger.info(f"Saving network to: {output_file}")
    save_network(network, output_file)
    logger.info("TES-SSRC system integration complete")


if __name__ == "__main__":
    main()





    
