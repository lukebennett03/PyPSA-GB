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

from astroid.nodes import Raise
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
    if total_capacity != new_total:
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
        network.add("Link", f"TES_{cell["charge"]}",
                    bus0=steam_bus,
                    bus1=tes_bus,
                    carrier=resolved_carriers["charge"],
                    efficiency=params["charge_efficiency"],
                    p_nom=params["charge_p_nom_mw_th"],)

        # Build TES storage
        network.add("Store", f"TES_{cell["store"]}",
                    bus=tes_bus,
                    carrier=resolved_carriers["storage"],
                    e_nom=params["tes_e_nom_mwh_th"],
                    e_cyclic=True)

        # Build SSRC link (discharge)
        network.add("Link", f"SSRC_{cell["discharge"]}",
                    bus0=tes_bus,
                    bus1=elec_bus,
                    carrier=resolved_carriers["ssrc"],
                    efficiency=params["ssrc_efficiency"],
                    p_nom=params["ssrc_p_nom_mw_th"],)


    logger.info(f"Added {len(storage_units)} TES charging links, TES storage and SSRC links")

def _validate(
        network: pypsa.Network,
):
    """
    This function will be used to validate final modular setup. Staged for development
    Args:
        network:

    Returns:

    """


# =============================================================================
# BUILD MODULES
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

    carriers = tes_config.get("carriers", {})
    steam_carrier = carriers.get("steam", "nuclear_steam")
    storage_carrier = carriers.get("storage", "nuclear_tes")

    logger.info(f"Building {modules_per_reactor} modules of "
                f"{len(storage_units)} units at {flex_generator}")

    # -------------------------------------------------------------------------
    # Extract key info about target generator
    # -------------------------------------------------------------------------
    target_generator_dict = _resolve_target(
        network,
        target_generator=flex_generator
    )

    # -------------------------------------------------------------------------
    # Extract module names
    # -------------------------------------------------------------------------
    names = _names(
        gen=flex_generator,
        modules=modules_per_reactor,
        units=storage_units,
    )

    # -------------------------------------------------------------------------
    # Build and extract carriers
    # -------------------------------------------------------------------------
    carriers = _add_carrier(
        network=network,
        cfg=tes_config,
    )

    # -------------------------------------------------------------------------
    # Build all busses and extract the steam bus
    # -------------------------------------------------------------------------
    steam_bus = _add_busses(
        network=network,
        target_dict=target_generator_dict,
        carriers=carriers,
        modules=modules_per_reactor,
        units=storage_units,
    )

    # -------------------------------------------------------------------------
    # Re-configure nuclear reactors
    # -------------------------------------------------------------------------
    new_nuclear_summary = _handle_fes_nuclear(
        network=network,
        target_dict=target_generator_dict,
        cfg=tes_config,
    )

    # -------------------------------------------------------------------------
    # Build reactor and PSRC
    # -------------------------------------------------------------------------
    _add_reactor_psrc(
        network=network,
        steam_bus=steam_bus,
        resolved_carriers=carriers,
        cfg=tes_config,
        target_dict=target_generator_dict,
    )

    # -------------------------------------------------------------------------
    # Build the modules
    # -------------------------------------------------------------------------
    for module in range(1, modules_per_reactor + 1):
        _add_module(
            network=network,
            target_dict=target_generator_dict,
            module=module,
            storage_units=storage_units,
            steam_bus=steam_bus,
            resolved_carriers=carriers
        )

    logger.info("Built TES-SSRC modules")

    return network

    
