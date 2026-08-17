"""
Nuclear TES-SSRC Integration Rules for PyPSA-GB

Couples a designated nuclear reactor to a modular thermal energy storage and
secondary steam Rankine cycle system, after Al Kindi et al. (2022, 2023).

Pipeline Stages:
  1. add_tes_ssrc - designate a reactor as an EPR, rescale the remaining
     nuclear fleet, and build the steam header, thermal stores and the
     primary and secondary conversion links

Inputs:
  - Complete network (.nc): all components integrated, clustered if enabled

Outputs:
  - Network (.nc): as input, plus the TES-SSRC system on the target reactor

See Also:
  - hydrogen.smk - same multi-carrier pattern (bus, links, store)
  - solve.smk - finalize_network consumes this rule's output
"""

# ══════════════════════════════════════════════════════════════════════════════
# RULES
# ══════════════════════════════════════════════════════════════════════════════

rule add_tes_ssrc:
    """
    Couple a modular TES-SSRC system to a designated nuclear reactor.

    The target FES nuclear generator is replaced by an EPR whose thermal output
    is held constant. Electricity reaches the grid through a primary steam
    Rankine cycle, while steam diverted to phase-change thermal stores is
    recovered through secondary cycles during high demand. The remaining
    nuclear fleet is rescaled so total system nuclear capacity is unchanged.

    Runs after clustering so the new buses attach to the final bus set. Inert
    when nuclear_tes.enabled is false, in which case the network passes through
    unchanged.

    Transforms: {scenario}_network_..._interconnectors.nc
                → {scenario}_network_nucleartes.nc

    Inputs:
      - network: complete network, clustered if enabled (.nc)

    Outputs:
      - network: network with TES-SSRC components added (.nc)

    Performance: ~10s for Reduced network
    """
    input:
        network=lambda wildcards: (
            _clustered_network_output(wildcards.scenario)
            if _is_clustering_enabled(wildcards.scenario)
            else f"{resources_path}/network/{wildcards.scenario}"
                 f"_network_demand_renewables_thermal_generators_storage"
                 f"_hydrogen_interconnectors.nc"
        )
    output:
        network=f"{resources_path}/network/{{scenario}}_network_nucleartes.nc"
    params:
        scenario=lambda wc: wc.scenario,
        nuclear_tes=lambda wc: scenarios[wc.scenario].get("nuclear_tes", {})
    wildcard_constraints:
        scenario=SCENARIO_REGEX
    log:
        "logs/nuclear/add_tes_ssrc_{scenario}.log"
    benchmark:
        "benchmarks/nuclear/add_tes_ssrc_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/nuclear/add_tes_ssrc.py"
