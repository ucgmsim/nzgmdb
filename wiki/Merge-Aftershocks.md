# 🔗 Merge Aftershocks

This step in the NZGMDB pipeline calculates aftershock flags and cluster labels for earthquakes using the Abrahamson & Woodell 2014/2018 Distance-Window with centroid Joyner-Boore distance (CRJB). The process identifies mainshock-aftershock relationships and assigns cluster labels to earthquake sequences.

---

## 🚀 Entry Point

To calculate aftershock flags for the earthquake source table, run the following Python script:

```bash
python -m nzgmdb.scripts.run_nzgmdb calculate-aftershocks <main_dir>
```

- **<main_dir>** is the top-level output directory where NZGMDB stores its results.

Example:
```bash
python -m nzgmdb.scripts.run_nzgmdb calculate-aftershocks nzgmdb_output/
```

This will analyze the earthquake source table and generate aftershock classifications based on spatial and temporal proximity criteria.

---

## 📋 Prerequisites

The Merge Aftershocks step requires the following inputs from previous pipeline steps:
- **[Calculate Distances](https://github.com/ucgmsim/nzgmdb/wiki/Calculate-Distances)** (generates the earthquake source table with distance calculations as well as the geometry table containing rupture polygons)

---

## ⚙️ Process

### 🔹 Rupture Polygon Generation

The algorithm creates rupture area polygons for each earthquake using two approaches:

#### **SRF-based Polygons (Preferred)**
For earthquakes with available Surface Rupture Format (SRF) files:
1. **Load SRF model** from the data registry
2. **Generate convex hull** from SRF geometry points
3. **Transform coordinates** from NZTM to WGS84 using pyproj
4. **Create polygon** from transformed hull coordinates

#### **Corner-based Polygons (Fallback)**
For earthquakes without SRF files:
1. **Extract corner coordinates** from geometry table (corner_0 through corner_3)
2. **Apply longitude normalization** using modulo 360 to handle negative values
3. **Create rectangular polygon** from corner points in WGS84 coordinates

### 🔹 ABWD-CRJB Classification Algorithm

The core aftershock detection follows the Abrahamson & Woodell methodology:

#### **Temporal Window Calculation**
```python
# Gardner-Knopoff time window for M < 6.5
sw_time = 10^(0.5409 * mag - 0.547) / 364.75

# Modified time window for M ≥ 6.5  
sw_time = 10^(0.032 * mag + 2.7389) / 364.75
```

#### **Spatial Distance Computation**
1. **Calculate rupture centroids** for all earthquake polygons
2. **Resample polygon boundaries** at ~1 km resolution using great circle distances
3. **Compute CRJB distances** between earthquake centroids and resampled rupture boundaries
4. **Apply distance threshold** based on configurable `crjb_cutoff` values

#### **Clustering Process**
1. **Sort earthquakes** by magnitude in descending order (largest first)
2. **Iterate through events** starting with the largest magnitude
3. **Find temporal candidates** within the time window of each potential mainshock
4. **Calculate spatial distances** using CRJB methodology
5. **Classify aftershocks** for events within both time and distance thresholds
6. **Assign cluster labels** to mainshock-aftershock sequences

### 🔹 Multi-threshold Analysis

The process runs the ABWD-CRJB algorithm for multiple distance cutoffs defined in the configuration:

| CRJB Cutoff | Description |
|-------------|-------------|
| 0 km | Zero-distance clustering (only direct overlaps) |
| 2 km | Very close spatial clustering |
| 5 km | Moderate spatial clustering |
| 10 km | Extended spatial clustering |

Each cutoff generates independent aftershock flags and cluster labels, allowing for sensitivity analysis of spatial thresholds.

---

## ⚙️ Configuration Parameters

Key configuration values from `config.yaml`:

| Parameter | Default Values | Description |
|-----------|----------------|-------------|
| `crjb_cutoffs` | [0, 2, 5, 10] | Distance thresholds in kilometers for spatial clustering |
| `ll_num` | 4326 | WGS84 coordinate system identifier |
| `nztm_num` | 2193 | New Zealand Transverse Mercator projection |

---

## 📦 Output

### 🔹 Enhanced Earthquake Source Table

The primary output is an updated earthquake source table with additional aftershock classification columns:

**File**: `flatfiles/earthquake_source_table_aftershocks.csv`

#### **Aftershock Flag Columns**
| Column | Description |
|--------|-------------|
| `aftershock_flag_crjb0` | Binary flag (0=mainshock, 1=aftershock) for 0 km cutoff |
| `aftershock_flag_crjb2` | Binary flag (0=mainshock, 1=aftershock) for 2 km cutoff |
| `aftershock_flag_crjb5` | Binary flag (0=mainshock, 1=aftershock) for 5 km cutoff |
| `aftershock_flag_crjb10` | Binary flag (0=mainshock, 1=aftershock) for 10 km cutoff |

#### **Cluster Label Columns**
| Column | Description |
|--------|-------------|
| `cluster_flag_crjb0` | Cluster identifier for 0 km cutoff (0=isolated event) |
| `cluster_flag_crjb2` | Cluster identifier for 2 km cutoff (0=isolated event) |
| `cluster_flag_crjb5` | Cluster identifier for 5 km cutoff (0=isolated event) |
| `cluster_flag_crjb10` | Cluster identifier for 10 km cutoff (0=isolated event) |

#### **Interpretation Guide**
- **Aftershock flag = 0**: Event classified as mainshock or isolated event
- **Aftershock flag = 1**: Event classified as aftershock to a larger magnitude event
- **Cluster flag = 0**: Event does not belong to any earthquake sequence
- **Cluster flag > 0**: Event belongs to earthquake sequence with the specified cluster ID

---

## 🔗 Related Steps

- **Previous**: [Calculate Distances](Calculate-Distances.md) - Provides the earthquake source table with distance calculations and geometry data required for Aftershock analysis
- **Next**: [Merge Flatfiles](Merge-Flatfiles.md) - Consolidates the aftershock-enhanced earthquake source table with other pipeline outputs for final database generation