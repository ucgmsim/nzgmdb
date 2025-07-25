# 📍 Fetching Site Table

This step in the NZGMDB pipeline collects metadata for all seismic stations within New Zealand. It combines station information from GeoNet with local metadata and assigns each site to a basin if applicable.

---


## 🚀 Entry Point

To generate the site table, run the following Python script:

```bash
python -m nzgmdb.scripts.run_nzgmdb generate-site-table-basin <main_dir>
```

- **<main_dir>** is the top-level output directory where NZGMDB stores its results.

Example:
```bash
python -m nzgmdb.scripts.run_nzgmdb generate-site-table-basin nzgmdb_output/
```

This will create the file:

```bash
nzgmdb_output/flatfiles/site_table_basin.csv
```

## ⚙️ Process

### 🔹 Fetch Station Metadata
- Uses ObsPy’s `FDSN_Client` to download station metadata from the GeoNet network, including:
  - Network code
  - Station code
  - Latitude / Longitude
  - Elevation

### 🔹 Merge with Local Site Metadata
- The fetched station list is merged with a local Geonet metadata summary CSV, which includes geotechnical properties such as:
  - NZS1170 Site Class
  - Vs30 median and standard deviation
  - Basin depths (Z1.0, Z2.5)
  - Other site parameters like T0 and Q factors
- ⚠️ **Note:** This metadata is loaded from the NZGMDB data registry and is **not** found directly from the GitHub repository.
- Stations missing from the metadata summary may have missing fields after the merge. These are handled later in the quality db step to filter these out and report which sites have missing data but have quality waveforms.

### 🔹 Assign Tectonic Domain
- Each site is assigned a tectonic domain number by spatially intersecting the site coordinates with a tectonic domain shapefile:
  - `TectonicDomains_Feb2021_8_NZTM.shp` (and associated `.dbf`, `.shx` files)
- The output field `site_domain_no` represents the tectonic domain ID indicating the regional tectonic environment of the station.

### 🔹 Add Basin Information

Adds basin information to the site table dataframe by checking if station coordinates fall inside known basin polygons.

- Loads the latest version of basin outlines from the velocity modelling repository.
- Uses a spatial point-in-polygon test to assign the basin name to sites located within basin boundaries.
- Adds a new column `basin` to the input dataframe.

---

## 📦 Output

- The final merged dataframe gets saved as `site_table.csv` in the flatfiles directory containing:

| Column          | Description                                                  |
|-----------------|--------------------------------------------------------------|
| `net`           | Network code (e.g., "NZ")                                    |
| `sta`           | Station code identifier (e.g., "DCZ")                        |
| `lat`           | Latitude of station location (WGS84)                         |
| `lon`           | Longitude of station location (WGS84)                        |
| `elev`          | Station elevation in meters                                  |
| `site_class`    | NZS1170 site classification (A is best, D is worst)          |
| `Vs30`          | Shear wave velocity (Vs30) value in m/s                      |
| `Vs30_std`      | Standard deviation of Vs30                                   |
| `Q_Vs30`        | Quality flag for Vs30                                        |
| `Vs30_Ref`      | Reference for Vs30 data source                               |
| `T0`            | Fundamental site period                                      |
| `T0_std`        | Standard deviation of T0                                     |
| `Q_T0`          | Quality flag for T0                                          |
| `D_T0`          | Data Source for T0                                           |
| `T0_ref`        | Reference for T0 data source                                 |
| `Z1.0`          | Basin depth to 1.0 km/s shear wave velocity (meters)         |
| `Z1.0_std`      | Standard deviation of Z1.0                                   |
| `Q_Z1.0`        | Quality flag for Z1.0                                        |
| `Z1.0_ref`      | Reference for Z1.0 data source                               |
| `Z2.5`          | Basin depth to 2.5 km/s shear wave velocity (meters, scaled) |
| `Z2.5_std`      | Standard deviation of Z2.5                                   |
| `Q_Z2.5`        | Quality flag for Z2.5                                        |
| `Z2.5_ref`      | Reference for Z2.5 data source                               |
| `site_domain_no`| Integer tectonic domain ID assigned from shapefile           |
| `basin`         | Basin name assigned based on spatial polygon test            |

---

## 🔗 Related Steps

- **Next**: [Parse GeoNet](Parse-Geonet.md) - Parses GeoNet data to extract waveform files and metadata