"""
Used in the intermediate stage after building network before adding TES-SSRC.

This code is split into two parts. First is finding all reactor names assigned
by PyPSA-GB as the model constructs these to meet FES capacities. Second we
locate the cluster corresponding to Bramford as this is the closest geographical
location to Sizewell B, giving a realistic anchor.

This is v1 devlopment. We will later make this more secure. For example Bramford may not exist.
We also may want to automate pulling scenario names instead of updatting by hand.
"""
from pathlib import Path

import pandas as pd
import pypsa

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = ["counterfactual", "tes5", "20_clusters_counterfactual"]

SCENARIO_PREFIX = "HT37"

for s in SCENARIOS:
     check_path = REPO / f"resources/network/{SCENARIO_PREFIX}_{s}_network_clustered_demand_renewables"
     "_thermal_generators_storage_hydrogen_interconnectors.nc"

     # Load network
     n = pypsa.Network(check_path)

     # Get nuclear reactors
     nuc = n.generators[n.generators.carrier.str.casefold() == "nuclear"]
     print(nuc[["bus", "carrier", "p_nom"]])
     print("nuclear total:", nuc.p_nom.sum())
     print("buses:", len(n.buses), "gens:", len(n.generators), "storage:", len(n.storage_units))

     # Check clusters
     cluster_path = REPO / f"resources/network/{SCENARIO_PREFIX}_{s}_clustering_busmap.csv"
     busmap = pd.read_csv(cluster_path)

     # Locate Bramford
     print(busmap[busmap.bus_id.str.contains("Bramford", case=False)])
