"""
Unit tests for the TES-SSRC builder helpers.

These run against a small synthetic network - no data files, no Snakemake,
no .nc. Run from the repository root:

    pytest tests/unit/test_add_tes_ssrc.py -v
"""

import pytest
import pypsa

from scripts.nuclear.add_tes_ssrc import (
    _add_busses,
    _add_carrier,
    _add_module,
    _add_reactor_psrc,
    _cell_names,
    _handle_fes_nuclear,
    _names,
    _resolve_target,
    load_tes_config,
)


# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture
def network() -> pypsa.Network:
    """
    Minimal network carrying several nuclear generators plus a non-nuclear
    generator sharing a bus with the target.

    Capacities mirror the FES HT35 reduced allocation so the numbers are
    recognisable. The target is deliberately not the first generator added.
    """
    n = pypsa.Network()
    n.set_snapshots(range(4))

    n.add("Carrier", "AC")
    n.add("Carrier", "nuclear")
    n.add("Carrier", "CCGT")

    n.add("Bus", "Keadby", carrier="AC", x=-0.75, y=53.60, v_nom=400)
    n.add("Bus", "Melksham", carrier="AC", x=-2.14, y=51.37, v_nom=400)
    n.add("Bus", "Bramford", carrier="AC", x=1.06, y=52.06, v_nom=400)

    # Target is second, so a loop that raises on the first mismatch fails here
    n.add("Generator", "FES_nuclear_NGET_Keadby",
          bus="Keadby", carrier="nuclear", p_nom=377.36)
    n.add("Generator", "FES_nuclear_NGET_Melksham",
          bus="Melksham", carrier="nuclear", p_nom=251.57)
    n.add("Generator", "FES_nuclear_NGET_Bramford",
          bus="Bramford", carrier="nuclear", p_nom=188.68)

    # Stands in for the remaining 17 FES nuclear allocations, bringing the
    # fleet to HT35's 5000 MW. Without it the fleet is smaller than a single
    # EPR and the compensation guard in _handle_fes_nuclear cannot be
    # satisfied - the same condition the five-reactor sensitivity will hit.
    n.add("Generator", "FES_nuclear_NGET_Rest",
          bus="Keadby", carrier="nuclear", p_nom=4182.39)

    # Non-nuclear generator at the same bus, to catch carrier-blind matching
    n.add("Generator", "CCGT_Melksham",
          bus="Melksham", carrier="CCGT", p_nom=500.0)

    n.add("Load", "demand", bus="Melksham", p_set=1000.0)

    return n


@pytest.fixture
def cfg() -> dict:
    """Minimal nuclear_tes config block."""
    return {
        "enabled": True,
        "target_generator": "FES_nuclear_NGET_Melksham",
    }


# =============================================================================
# _resolve_target
# =============================================================================
def test_returns_all_expected_fields(network, cfg):
    """Every key the downstream builder relies on must be populated."""
    spec = _resolve_target(network, cfg["target_generator"])

    for key in ("name", "electrical_bus", "bus_x", "bus_y", "p_nom"):
        assert key in spec, f"missing key: {key}"
        assert spec[key] is not None, f"key left unpopulated: {key}"


def test_resolves_correct_generator(network, cfg):
    """Name, bus and capacity come from the target, not another generator."""
    spec = _resolve_target(network, cfg["target_generator"])

    assert spec["name"] == "FES_nuclear_NGET_Melksham"
    assert spec["electrical_bus"] == "Melksham"
    assert spec["p_nom"] == pytest.approx(251.57)


def test_resolves_when_target_is_not_first(network, cfg):
    """
    The target is the second generator in the network. A loop that raises on
    the first non-matching generator fails this test.
    """
    spec = _resolve_target(network, cfg["target_generator"])
    assert spec["name"] == "FES_nuclear_NGET_Melksham"


def test_returns_bus_coordinates(network, cfg):
    """
    x and y are needed for the steam and TES buses. Without them the new
    buses carry NaN coordinates and spatial plotting breaks.
    """
    spec = _resolve_target(network, cfg["target_generator"])

    assert spec["bus_x"] == pytest.approx(-2.14)
    assert spec["bus_y"] == pytest.approx(51.37)


def test_ignores_non_nuclear_generators(network, cfg):
    """
    A CCGT sits on the same bus. Matching must be restricted to the nuclear
    carrier, or a differently-named plant could be selected.
    """
    cfg = {**cfg, "target_generator": "CCGT_Melksham"}

    with pytest.raises(ValueError):
        _resolve_target(network, cfg["target_generator"])


def test_raises_on_unknown_generator(network, cfg):
    """An unmatched target must fail loudly, not return an empty spec."""
    cfg = {**cfg, "target_generator": "FES_nuclear_NGET_Nowhere"}

    with pytest.raises(ValueError):
        _resolve_target(network, cfg["target_generator"])


def test_raises_on_ambiguous_match(network, cfg):
    """
    Only relevant if matching is by substring. 'FES_nuclear' matches three
    generators, and silently picking one would site the reactor arbitrarily.

    Delete this test if you settle on exact-name matching.
    """
    cfg = {**cfg, "target_generator": "FES_nuclear"}

    with pytest.raises(ValueError):
        _resolve_target(network, cfg["target_generator"])


def test_resolves_a_second_generator(network):
    """
    Resolution must work for any nuclear generator, not just the fixture
    default - the target is a config choice and will change across scenarios.
    """
    spec = _resolve_target(network, "FES_nuclear_NGET_Bramford")

    assert spec["electrical_bus"] == "Bramford"
    assert spec["p_nom"] == pytest.approx(188.68)


def test_does_not_mutate_network(network, cfg):
    """Resolution is a pure lookup - it must not add or remove components."""
    before = (len(network.buses), len(network.generators), len(network.links))

    _resolve_target(network, cfg["target_generator"])

    after = (len(network.buses), len(network.generators), len(network.links))
    assert before == after


# =============================================================================
# _names
# =============================================================================
GEN = "FES_nuclear_NGET_Melksham"
UNITS = ["unit_1", "unit_2"]


def test_names_returns_one_entry_per_grid_cell():
    """5 modules x 2 units = 10 names in every category."""
    names = _names(GEN, modules=5, units=UNITS)

    for key, values in names.items():
        assert len(values) == 10, f"{key} has {len(values)} entries, expected 10"


def test_names_are_globally_unique():
    """
    The single most important property. PyPSA component names must be unique;
    duplicates silently collapse into one component rather than erroring.
    """
    names = _names(GEN, modules=5, units=UNITS)

    flat = [n for values in names.values() for n in values]
    assert len(set(flat)) == len(flat), "duplicate component names generated"


def test_names_encode_generator_module_and_unit():
    """A name must identify which cell it belongs to, for later selection."""
    names = _names(GEN, modules=2, units=UNITS)

    assert f"{GEN}_1_unit_1_store" in names["stores"]
    assert f"{GEN}_2_unit_2_charge" in names["charges"]


def test_names_includes_the_tes_bus():
    """
    The bus is the one name shared between the bus builder and the store and
    links attached to it. If it is not owned here, those callers compute it
    independently and can diverge.
    """
    names = _names(GEN, modules=5, units=UNITS)

    assert "buses" in names
    assert len(names["buses"]) == 10


def test_names_modules_start_at_one():
    """Modules are 1-indexed - a module 0 would read as the counterfactual."""
    names = _names(GEN, modules=1, units=UNITS)

    assert not any("_0_" in n for n in names["stores"])


def test_names_zero_modules_returns_empty_lists():
    """
    modules_per_reactor: 0 is the counterfactual. It must return empty lists
    rather than raising, so the builder adds reactor and PSRC only.
    """
    names = _names(GEN, modules=0, units=UNITS)

    for key, values in names.items():
        assert values == [], f"{key} should be empty for 0 modules"


def test_names_are_deterministic():
    """Two calls must agree, or components created in different places diverge."""
    assert _names(GEN, 3, UNITS) == _names(GEN, 3, UNITS)


def test_names_distinguish_units():
    """unit_1 and unit_2 differ in SSRC efficiency, so must be separable."""
    names = _names(GEN, modules=1, units=UNITS)

    assert len({n for n in names["stores"]}) == 2


# =============================================================================
# _cell_names
# =============================================================================
def test_cell_names_returns_all_four_components():
    """One storage train needs a bus, a store, and two links."""
    cell = _cell_names(GEN, module=1, unit="unit_1")

    assert set(cell) == {"bus", "store", "charge", "discharge"}


def test_cell_names_share_a_stem():
    """
    All four names derive from one stem, so a component can be traced back to
    its cell by string alone.
    """
    cell = _cell_names(GEN, module=3, unit="unit_2")
    stem = f"{GEN}_3_unit_2"

    assert all(name.startswith(stem) for name in cell.values())


def test_cell_names_agree_with_grid_names():
    """
    The grid wrapper must produce exactly what the per-cell function does.
    If these drift, the builder and the validation disagree about what
    exists in the network.
    """
    grid = _names(GEN, modules=2, units=UNITS)

    expected_buses = [
        _cell_names(GEN, module, unit)["bus"]
        for module in (1, 2)
        for unit in UNITS
    ]

    assert grid["buses"] == expected_buses


def test_cell_names_are_unique_within_a_cell():
    """The four names must differ from each other, not just across cells."""
    cell = _cell_names(GEN, module=1, unit="unit_1")

    assert len(set(cell.values())) == 4


# =============================================================================
# SHARED FIXTURES FOR THE BUILDER HELPERS
# =============================================================================
@pytest.fixture
def tes_cfg() -> dict:
    """
    The real nuclear_tes block from defaults.yaml, retargeted at the fixture
    generator. Using the shipped config means a typo there fails here rather
    than on HT35.
    """
    cfg = load_tes_config()
    return {**cfg, "enabled": True, "target_generator": GEN}


@pytest.fixture
def carriers(network, tes_cfg) -> dict:
    """Carriers declared, returning the role-to-name mapping."""
    return _add_carrier(network, tes_cfg)


@pytest.fixture
def target(network, tes_cfg) -> dict:
    """Resolved target spec."""
    return _resolve_target(network, tes_cfg["target_generator"])


# =============================================================================
# _add_carrier
# =============================================================================
def test_carriers_added_to_network(network, tes_cfg):
    """All five roles must exist as carriers before components reference them."""
    resolved = _add_carrier(network, tes_cfg)

    for name in resolved.values():
        assert name in network.carriers.index, f"carrier not added: {name}"


def test_carriers_returns_role_mapping(network, tes_cfg):
    """Downstream helpers select carriers by role, not by name."""
    resolved = _add_carrier(network, tes_cfg)

    assert set(resolved) == {"steam", "storage", "psrc", "charge", "ssrc"}


def test_carriers_have_distinct_colours(network, tes_cfg):
    """Shared colours make steam and storage indistinguishable in plots."""
    resolved = _add_carrier(network, tes_cfg)

    colours = network.carriers.loc[list(resolved.values()), "color"]
    assert len(set(colours)) == len(colours)


def test_carriers_are_idempotent(network, tes_cfg):
    """Re-running the builder must not duplicate or overwrite carriers."""
    first = _add_carrier(network, tes_cfg)
    count = len(network.carriers)

    second = _add_carrier(network, tes_cfg)

    assert first == second
    assert len(network.carriers) == count


def test_carriers_do_not_touch_existing(network, tes_cfg):
    """An unrelated carrier already in the network must be left alone."""
    before = network.carriers.loc["nuclear"].to_dict()

    _add_carrier(network, tes_cfg)

    assert network.carriers.loc["nuclear"].to_dict() == before


# =============================================================================
# _add_busses
# =============================================================================
def test_busses_created_for_every_cell(network, target, carriers):
    """One steam header plus one store bus per module-unit cell."""
    before = len(network.buses)

    _add_busses(network, target, carriers, modules=5, units=UNITS)

    assert len(network.buses) == before + 1 + (5 * len(UNITS))


def test_busses_returns_steam_bus(network, target, carriers):
    """The steam bus name is needed by the PSRC and charge links."""
    steam_bus = _add_busses(network, target, carriers, modules=5, units=UNITS)

    assert steam_bus in network.buses.index
    assert network.buses.at[steam_bus, "carrier"] == carriers["steam"]


def test_bus_names_match_cell_names(network, target, carriers):
    """
    The buses created here must be exactly those _cell_names reports, or the
    stores and links attached later will reference buses that do not exist.
    """
    _add_busses(network, target, carriers, modules=3, units=UNITS)

    expected = _names(GEN, modules=3, units=UNITS)["buses"]
    for bus in expected:
        assert bus in network.buses.index, f"bus missing: {bus}"


def test_busses_inherit_electrical_coordinates(network, target, carriers):
    """
    New buses take the parent bus coordinates so spatial clustering keeps them
    with their reactor. NaN coordinates break plotting.
    """
    _add_busses(network, target, carriers, modules=2, units=UNITS)

    new = network.buses[network.buses.carrier.isin(
        [carriers["steam"], carriers["storage"]]
    )]

    assert new["x"].notna().all()
    assert new["y"].notna().all()

    # Copied verbatim from the parent bus, so exact equality is expected.
    # Comparing a Series to pytest.approx collapses to a single bool rather
    # than testing elementwise.
    assert (new["x"] == target["bus_x"]).all()
    assert (new["y"] == target["bus_y"]).all()


def test_busses_zero_modules_adds_steam_only(network, target, carriers):
    """The counterfactual needs the steam header but no stores."""
    before = len(network.buses)

    _add_busses(network, target, carriers, modules=0, units=UNITS)

    assert len(network.buses) == before + 1


# =============================================================================
# _handle_fes_nuclear
# =============================================================================
def test_fes_removes_target_generator(network, target, tes_cfg):
    """The target must go, or it generates alongside the PSRC."""
    _handle_fes_nuclear(network, target, tes_cfg)

    assert GEN not in network.generators.index


def test_fes_preserves_total_nuclear_capacity(network, target, tes_cfg):
    """
    The whole purpose of the rescaling. Replacing a 251 MW allocation with a
    1610 MW EPR would otherwise raise GB nuclear by over 1 GW.
    """
    nuclear = network.generators.carrier.str.casefold().eq("nuclear")
    before = network.generators.loc[nuclear, "p_nom"].sum()

    summary = _handle_fes_nuclear(network, target, tes_cfg)

    remaining = network.generators.carrier.str.casefold().eq("nuclear")
    after = network.generators.loc[remaining, "p_nom"].sum()
    epr = tes_cfg["reactor_p_nom_th"] * tes_cfg["psrc_efficiency"]

    assert after + epr == pytest.approx(before)
    assert summary["total_before"] == pytest.approx(before)


def test_fes_leaves_non_nuclear_alone(network, target, tes_cfg):
    """Rescaling must not touch the CCGT."""
    before = network.generators.at["CCGT_Melksham", "p_nom"]

    _handle_fes_nuclear(network, target, tes_cfg)

    assert network.generators.at["CCGT_Melksham", "p_nom"] == before


def test_fes_scale_factor_is_below_one(network, target, tes_cfg):
    """The EPR is larger than the generator it replaces, so others shrink."""
    summary = _handle_fes_nuclear(network, target, tes_cfg)

    assert 0 < summary["scale_factor"] < 1


def test_fes_raises_when_epr_exceeds_fleet(network, target, tes_cfg):
    """
    No positive scale factor exists if the EPR is larger than total nuclear.
    This is what the five-reactor sensitivity will hit.
    """
    cfg = {**tes_cfg, "reactor_p_nom_th": 100_000}

    with pytest.raises(ValueError):
        _handle_fes_nuclear(network, target, cfg)


# =============================================================================
# _add_reactor_psrc
# =============================================================================
@pytest.fixture
def prepared(network, target, carriers, tes_cfg):
    """Network with buses added and the FES generator handled."""
    steam_bus = _add_busses(network, target, carriers, modules=5, units=UNITS)
    _handle_fes_nuclear(network, target, tes_cfg)
    return steam_bus


def test_reactor_has_constant_thermal_output(network, prepared, carriers,
                                             tes_cfg, target):
    """
    p_min_pu == p_max_pu == 1.0 is what makes this storage-coupled rather than
    direct reactor flexing. Without it the optimiser throttles the reactor.
    """
    _add_reactor_psrc(network, prepared, carriers, tes_cfg, target)

    reactor = f"{GEN}_reactor"
    assert network.generators.at[reactor, "p_min_pu"] == 1.0
    assert network.generators.at[reactor, "p_max_pu"] == 1.0
    assert network.generators.at[reactor, "p_nom"] == tes_cfg["reactor_p_nom_th"]


def test_reactor_carrier_is_not_nuclear(network, prepared, carriers,
                                        tes_cfg, target):
    """
    If the reactor carried 'nuclear' the fleet rescaling would shrink it on
    any subsequent run.
    """
    _add_reactor_psrc(network, prepared, carriers, tes_cfg, target)

    assert network.generators.at[f"{GEN}_reactor", "carrier"] != "nuclear"


def test_psrc_link_connects_steam_to_electrical(network, prepared, carriers,
                                                tes_cfg, target):
    """p_nom is thermal and measured at bus0; output is p_nom x efficiency."""
    _add_reactor_psrc(network, prepared, carriers, tes_cfg, target)

    psrc = f"{GEN}_psrc"
    assert network.links.at[psrc, "bus0"] == prepared
    assert network.links.at[psrc, "bus1"] == target["electrical_bus"]
    assert network.links.at[psrc, "p_nom"] == tes_cfg["reactor_p_nom_th"]


def test_psrc_delivers_expected_electrical_capacity(network, prepared, carriers,
                                                    tes_cfg, target):
    """4520 MW_th x 0.356 should give the 1610 MW_e of the source EPR."""
    _add_reactor_psrc(network, prepared, carriers, tes_cfg, target)

    psrc = network.links.loc[f"{GEN}_psrc"]
    assert psrc["p_nom"] * psrc["efficiency"] == pytest.approx(1610, rel=1e-3)


def test_reactor_raises_if_fes_generator_still_present(network, target,
                                                       carriers, tes_cfg):
    """The precondition guard - ordering must be enforced, not assumed."""
    steam_bus = _add_busses(network, target, carriers, modules=1, units=UNITS)

    with pytest.raises(RuntimeError):
        _add_reactor_psrc(network, steam_bus, carriers, tes_cfg, target)


# =============================================================================
# _add_module
# =============================================================================
def test_module_adds_three_components_per_unit(network, prepared, carriers,
                                               tes_cfg, target):
    """Each storage train is a charge link, a store and an SSRC link."""
    units = tes_cfg["storage_units"]
    before_links, before_stores = len(network.links), len(network.stores)

    _add_module(network, target, 1, units, prepared, carriers)

    assert len(network.links) == before_links + 2 * len(units)
    assert len(network.stores) == before_stores + len(units)


def test_module_components_reference_existing_buses(network, prepared, carriers,
                                                    tes_cfg, target):
    """
    Catches divergence between _add_busses and _add_module. PyPSA will create
    a link to a nonexistent bus without complaint; it fails only at solve time.
    """
    _add_module(network, target, 1, tes_cfg["storage_units"], prepared, carriers)

    for bus in network.stores["bus"]:
        assert bus in network.buses.index, f"store on missing bus: {bus}"

    for col in ("bus0", "bus1"):
        for bus in network.links[col]:
            assert bus in network.buses.index, f"link on missing bus: {bus}"


def test_module_stores_are_cyclic(network, prepared, carriers, tes_cfg, target):
    """
    Without e_cyclic the optimiser can drain the stores over the horizon and
    treat the stored heat as free, flattering the flexible scenario.
    """
    _add_module(network, target, 1, tes_cfg["storage_units"], prepared, carriers)

    assert network.stores["e_cyclic"].all()


def test_module_ssrc_capacity_matches_source(network, prepared, carriers,
                                             tes_cfg, target):
    """
    Five modules should deliver 519.7 MW_e, giving the 2130 MW plant maximum
    reported by Al Kindi et al.
    """
    units = tes_cfg["storage_units"]
    for module in range(1, tes_cfg["modules_per_reactor"] + 1):
        _add_module(network, target, module, units, prepared, carriers)

    ssrc = network.links[network.links.carrier == carriers["ssrc"]]
    added = (ssrc["p_nom"] * ssrc["efficiency"]).sum()

    assert added == pytest.approx(519.7, abs=1.0)


def test_module_names_are_unique_across_modules(network, prepared, carriers,
                                                tes_cfg, target):
    """
    Five modules must give ten distinct trains. Duplicate names silently
    collapse into one component rather than raising.
    """
    units = tes_cfg["storage_units"]
    for module in range(1, 6):
        _add_module(network, target, module, units, prepared, carriers)

    assert len(network.stores) == 10
    assert len(set(network.stores.index)) == 10


def test_network_is_consistent(network, prepared, carriers, tes_cfg, target):
    """
    PyPSA's own structural check - dangling bus references, unknown carriers,
    missing attributes. Cheaper than rediscovering these at solve time.
    """
    _add_reactor_psrc(network, prepared, carriers, tes_cfg, target)

    units = tes_cfg["storage_units"]
    for module in range(1, tes_cfg["modules_per_reactor"] + 1):
        _add_module(network, target, module, units, prepared, carriers)

    network.consistency_check()
