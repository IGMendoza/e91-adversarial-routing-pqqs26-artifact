# Topology Bench (Real Topologies)

Local copy of the Topology Bench real-topology dataset for physical-topology
corpus ingestion.

Source page: https://topologybench.github.io
Zenodo record: https://zenodo.org/records/13921775

Downloaded archive:

```text
raw/real_topologies.zip
```

Archive SHA-256:

```text
f2ad31ecc53378dd8992beedfe93141454bc1915257bd78e4e0dca3484ed45c8
```

Extracted workbooks:

```text
extracted/TOP_<n>_<NAME>.xlsx
```

The archive contains 105 real core optical network topologies, each as an
`.xlsx` workbook with a `Nodes_*` sheet (`Node_ID`, `Latitude`, `Longitude`,
location metadata) and an `Edges_*` sheet (`Edge_ID`, `Source`, `Destination`,
`Computed Length (km)`). These are documented real fibre/core-network
topologies with geospatial link lengths, so corpus ingestion labels them
`topology_bench`, `telecom_topology`, and `physical_length` edge semantics
(lengths converted km -> m at ingest).

Citation:

```bibtex
@article{matzner2024topology,
  title={Topology Bench: systematic graph-based benchmarking for core optical networks},
  author={Matzner, Robin and Ahuja, Akanksha and Sadeghi, Rasoul and Doherty, Michael and Beghelli, Alejandra and Savory, Seb J and Bayvel, Polina},
  journal={Journal of Optical Communications and Networking},
  volume={17}, number={1}, pages={7--27}, year={2024}, publisher={IEEE}
}
```

The raw and extracted dataset files are ignored by git.
