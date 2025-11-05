# 🌋 Add Tectonic Domain

This step in the NZGMDB pipeline refines earthquake source metadata by incorporating tectonic classification and applying enhanced relocation techniques.
Specifically, it performs **relocations** and **tectonic mapping** by integrating multiple earthquake catalog sources to generate the most accurate event locations and tectonic classifications.

---

## 🚀 Entry Point

To add tectonic domains to the earthquake source table, run:

```bash
python -m nzgmdb.scripts.run_nzgmdb merge-tect-domain <eq_source_ffp> <output_dir> [--n-procs N]
```

**Parameters:**
- **eq_source_ffp**: Path to the earthquake source table CSV file (from [Parse Geonet](https://github.com/ucgmsim/nzgmdb/wiki/Parse-Geonet) step)
- **output_dir**: Directory where the enhanced earthquake source table will be saved
- **n-procs**: Number of processes to use for parallel computation (default: 1)

**Example:**
```bash
python -m nzgmdb.scripts.run_nzgmdb merge-tect-domain \
    nzgmdb_output/flatfiles/earthquake_source_table_geonet.csv \
    nzgmdb_output/flatfiles/ \
    --n-procs 6
```

This creates the file:
```bash
nzgmdb_output/flatfiles/earthquake_source_table_tectonic.csv
```

---

## ⚙️ Process Overview

The Add Tectonic Domain step performs **four main functions** in sequence:

### 🔹 1. Event Relocations
Relocations are applied from the Reyners Catalogue to improve location accuracy:

**[Reyners Catalogue Relocations](https://www.researchgate.net/publication/235997841_Tracking_repeated_subduction_of_the_Hikurangi_Plateau_beneath_New_Zealand)**
- Merges relocations from the Reyners Catalogue for improved earthquake locations
- Updates `lat`, `lon`, `depth`, `loc_type`, and `loc_grid` fields when better locations are available
- Sets `reloc` field to "reyners" for successfully relocated events

### 🔹 2. Centroid Moment Tensor (CMT) Solutions
GeoNet CMT location solutions are applied, which override Reyners relocations when overlapping:

**GeoNet CMT Solutions**

- Applies Centroid Moment Tensor (CMT) solutions from GeoNet
- Updates source parameters: mag (Mw), lat, lon, depth
- Sets metadata fields: mag_type="Mw", mag_method="CMT", loc_type="CMT", loc_grid="CMT", reloc="no"

### 🔹 3. Tectonic Classification
Each earthquake is classified using the **NGA-SUB 2020 methodology** combined with New Zealand-specific enhancements:

**Regional Fault Zone Analysis**
- **Hikurangi-Kermadec Subduction Zone**: Uses fault surface definitions with seismogenic zone depths (10-47 km)
- **Puysegur Subduction Zone**: Applies separate fault geometry with depths (11-30 km)

**NGA-SUB 2020 Classification System**
Events are classified into spatial regions and assigned tectonic types:

- **Region A (Offshore)**: 
  - depth < 60 km → "Outer-rise"
  - depth ≥ 60 km → "Slab"

- **Region B (Seismogenic Zone)**:
  - depth < min(slab surface, 20 km) → "Crustal"
  - depth between seismogenic zone bounds → "Interface"
  - depth > 60 km → "Slab"

- **Region C (Downdip)**:
  - depth < 30 km → "Crustal" 
  - depth > slab surface → "Slab"
  - elsewhere → "Undetermined"

- **Farfield**:
  - depth < 30 km → "Crustal"
  - depth > 30 km → "Undetermined"

**NZSMDB Integration**
- Merges tectonic classifications from the NZ Strong Motion Database
- NZSMDB classifications take precedence over NGA-SUB classifications when available
- Sets `tect_method="NZSMDB"` for expert-reviewed classifications

### 🔹 4. Neotectonic Domain Assignment
Each earthquake is assigned to a **tectonic domain** based on its geographic location:

**Spatial Domain Mapping**
- Uses `TectonicDomains_Feb2021_8_NZTM.shp` shapefile containing New Zealand's tectonic domain boundaries
- Converts coordinates from WGS84 to NZTM projection for accurate spatial analysis
- Handles both Polygon and MultiPolygon geometries using parallel point-in-polygon algorithms

---

## 📦 Output

The enhanced earthquake source table contains all original fields plus:

### **Relocation Fields**
| Field | Description |
|-------|-------------|
| `reloc` | Relocation source: "no", "reyners" |
| `mag_method` | Magnitude determination method (e.g., "CMT") |
| `loc_type` | Location determination method (e.g., "CMT") |
| `loc_grid` | Grid/network used for location |

### **Tectonic Classification Fields**
| Field | Description |
|-------|-------------|
| `tect_class` | Tectonic type: "Crustal", "Interface", "Slab", "Outer-rise", "Undetermined" |
| `tect_method` | Classification method |

### **Neotectonic Domain Fields**
| Field | Description |
|-------|-------------|
| `domain_no` | Numeric domain identifier (0 = Oceanic) |
| `domain_type` | Domain name (e.g., "Oceanic", "Canterbury") |

---

## 🔧 Configuration

The step uses several configuration parameters from `config.yaml`:

- **nzsmdb_url**: URL for NZ Strong Motion Database flatfile
- **cmt_url**: URL for GeoNet CMT solutions
- **ll_num**: WGS84 coordinate system identifier
- **nztm_num**: NZTM coordinate system identifier

---

## 🔗 Related Steps

- **Previous**: [Waveform Extraction](Waveform-Extraction.md) - Extracts waveform data from the FDSN for selected events and stations
- **Next**: [Phase Arrival](Phase-Arrival.md) - Uses PhaseNet to pick phase arrivals directly on waveforms
- **Related**: [Calculate Distances](Calculate-Distances.md) - Uses tectonic classifications to assist distance calculations