"""Registry for NZGMDB data files."""

import pooch

REGISTRY = {
    "hik_kerm_fault_300km_wgs84_poslon.txt": "sha256:1a199978b6c9c608f8473539b639a8825c1091167da3d14b07c7268528320e03",
    "Geonet_Metadata_Summary_v1.4.csv": "sha256:7884422c3fcae0810c02948ba1a3bd39ba5793ba28189e90d730541be1c207c0",
    "puy_slab2_dep_02.26.18.xyz": "sha256:9ebe4feab4ee3b80e3fe403f2d873f94e4d7f06d937d721cc9e154ecee83e3c0",
    "focal_mech_tectonic_domain_v1.csv": "sha256:1f1e0c4b7f9ca1b87fb2ca4883e587f330fe82b5bbf9ebb6eb8f4d12aa2e1936",
    "GeoNet_CMT_solutions_20201129_PreferredNodalPlane_v1.csv": "sha256:16596bfdbd0019bad22eed31a0cbeabba7d58d9e569a450ae4139222a473c296",
    "Mw_rrup.txt": "sha256:016cf1bd6d99278a8cda917596cf8d817ef5b6249daad9cf3d6c4d9cc224f204",
    "3366146.srf": "sha256:21fa6c9f1f729167dcd78020e80f94a726d22ec784b90f338fb28cf830c034a0",
    "3468575.srf": "sha256:85cd9aa1934fd7a50e00f0492d8bd1bb689e64e25110141eb6f30c467e2b8475",
    "2013p543824.srf": "sha256:eb2bfe6ae3aa13f3476b47d8a6e2fdcd47f9e9e0a9d216b58ae1702c8ef0c8bb",
    "2013p613797.srf": "sha256:9d20dd07e2e167f5abfe15382578bb804b4d8551c75d6e9b39a63601c32ab767",
    "2103645.srf": "sha256:c1868e64297f7d39aad6632b7bfc91ebd2a9767984531a9c08c730bf3768f1cb",
    "2016p661332.srf": "sha256:4790557b4eebe94e7080f0246d039887b1b26692055bae72c8499a09fb4978f5",
    "3124785.srf": "sha256:5f645afb072c437ef4378f80b11d2cc7fea04ddb0e0e6c294db48e52607a8e49",
    "3528839.srf": "sha256:741f81247007c319b97d51d52b8eeeb711f11e3322cca67d1094f20e738565b6",
    "3631359.srf": "sha256:e7b338624cdea0a250b8ada0434df16a370d1c9da03b413fb0fa9910891f7f90",
    "2016p858000.srf": "sha256:bfa9015bf9a18a432274f6052d2c80781c2cc6a48684f54793dd38f94db641e8",
    "TectonicDomains_Feb2021_8_NZTM.shp": "sha256:f4513bed118ebcc025819430f407375929cad823062944572870dc606981f8cb",
    "TectonicDomains_Feb2021_8_NZTM.shx": "sha256:9ca42e270e5604f4740b3d99ea6a72daa75eaae0e6c6bda3c4e5c86a72369403",
    "TectonicDomains_Feb2021_8_NZTM.dbf": "sha256:534c644c5d4d6a08106752e2fe33894bc820645efa017e0137ef5b8d44e8200c",
    "tectonic_domain_polygon_points.csv": "sha256:a54c30a4e68bb078c6ce9d99bd30f902a9627d57fd39392b403064de61b1f1a4",
}

URLS = {
    "focal_mech_tectonic_domain_v1.csv": "https://www.dropbox.com/scl/fi/zseg304cbjmti7gg5tdyv/focal_mech_tectonic_domain_v1.csv?rlkey=kfb9ttvnv9yi9zftixw6kmz4v&st=4j9pgpgj&dl=1",
    "Geonet_Metadata_Summary_v1.4.csv": "https://www.dropbox.com/scl/fi/iev7qmoqqzvc5quhf8mk8/Geonet-Metadata-Summary_v1.4.csv?rlkey=7twwwck5iy5zao7lwao6xodvm&st=6m3elzuu&dl=1",
    "GeoNet_CMT_solutions_20201129_PreferredNodalPlane_v1.csv": "https://www.dropbox.com/scl/fi/fq28jx8jlbozj0d1x5tnq/GeoNet_CMT_solutions_20201129_PreferredNodalPlane_v1.csv?rlkey=30xj6n7ara0vz4t8kg4pz8w5s&st=63x7nr3j&dl=1",
    "hik_kerm_fault_300km_wgs84_poslon.txt": "https://www.dropbox.com/scl/fi/ig3ajufpv4xg2qjfxxuup/hik_kerm_fault_300km_wgs84_poslon.txt?rlkey=9jajfkq2elrzwzzgh6px17k8e&st=6ham2oox&dl=1",
    "Mw_rrup.txt": "https://www.dropbox.com/scl/fi/e3o9v9ze9e4955xxtrl14/Mw_rrup.txt?rlkey=c663zntx7gaeyxt04i97r62nu&st=6ri3c620&dl=1",
    "puy_slab2_dep_02.26.18.xyz": "https://www.dropbox.com/scl/fi/mhxm77lnbnsmye8u6jd7o/puy_slab2_dep_02.26.18.xyz?rlkey=4qk0rfm6rfuzmvioswmq49luc&st=gdjpiy57&dl=1",
    "tectonic_domain_polygon_points.csv": "https://www.dropbox.com/scl/fi/tnwzajuyhco7mnwg3ja7o/tectonic_domain_polygon_points.csv?rlkey=hti6fmulztf784o9qzd17x4jb&st=etxrfpuy&dl=1",
    "TectonicDomains_Feb2021_8_NZTM.dbf": "https://www.dropbox.com/scl/fi/o2z83zbg3yvxq8yqiwkqk/TectonicDomains_Feb2021_8_NZTM.dbf?rlkey=o4imoxtoj8psa3ndvryqrx2qk&st=qvcwp7hj&dl=1",
    "TectonicDomains_Feb2021_8_NZTM.shp": "https://www.dropbox.com/scl/fi/69qwsa48gvtzx056vqirs/TectonicDomains_Feb2021_8_NZTM.shp?rlkey=qjkzra3xu0z9xjkb3dcji4max&st=kfsh8gaw&dl=1",
    "TectonicDomains_Feb2021_8_NZTM.shx": "https://www.dropbox.com/scl/fi/3qyw5v3cw0dzi41d6wnp8/TectonicDomains_Feb2021_8_NZTM.shx?rlkey=ducg84xqhi2ua6ku2qhbiutrg&st=flmk11e0&dl=1",
    "2013p543824.srf": "https://www.dropbox.com/scl/fi/013i860bt3os1k80leai5/2013p543824.srf?rlkey=ww3clc60gud4sycwfzmgopvix&st=jpu99tse&dl=1",
    "2013p613797.srf": "https://www.dropbox.com/scl/fi/7wx7glahf8cufo1g7umzc/2013p613797.srf?rlkey=1x9e5yb2nkf5468rs0bogm8ev&st=dut8d2uw&dl=1",
    "2016p661332.srf": "https://www.dropbox.com/scl/fi/pp7s6iqilrdgbrs00wi55/2016p661332.srf?rlkey=bhgcddk3bcdpxoycorlhp2in6&st=b8o2bk4w&dl=1",
    "2016p858000.srf": "https://www.dropbox.com/scl/fi/jpazrlo1ahev1agt0jue3/2016p858000.srf?rlkey=wkdvbqet5jxa4uckl7z403q87&st=yzapzvr6&dl=1",
    "2103645.srf": "https://www.dropbox.com/scl/fi/dihn6pug9ulaj5a414hjt/2103645.srf?rlkey=8uq2zo7c8roz1dtuw21wzw0z8&st=hf874irv&dl=1",
    "3124785.srf": "https://www.dropbox.com/scl/fi/5d7d6zmw9zlah00u2onmq/3124785.srf?rlkey=9swwgrifuemb4ri3tw0k2xt0a&st=n3h3g22p&dl=1",
    "3366146.srf": "https://www.dropbox.com/scl/fi/skrq7jp2lwn53mvrno9pv/3366146.srf?rlkey=rbu10agccqzuberpdm80peav3&st=lcqj4ijh&dl=1",
    "3468575.srf": "https://www.dropbox.com/scl/fi/rt2xedggp3yt6jyvmgqmx/3468575.srf?rlkey=t06fswe8kr415k68kc6y58m32&st=2wemmuzx&dl=1",
    "3528839.srf": "https://www.dropbox.com/scl/fi/4cps30pfrt6vw8a2ih1jr/3528839.srf?rlkey=tssde379h8lfn6ql0n29fl7dx&st=u2jnxrsx&dl=1",
    "3631359.srf": "https://www.dropbox.com/scl/fi/g66sx9d30r2k2xs9kg977/3631359.srf?rlkey=ne7zplqwd1datludrae7a3b2o&st=idq88bbj&dl=1",
}

NZGMDB_DATA = pooch.create(
    path=pooch.os_cache("nzgmdb_data"),
    base_url="",
    registry=REGISTRY,
    urls=URLS,
)
