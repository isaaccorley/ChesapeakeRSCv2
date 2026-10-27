"""Save NAIP footprints and acquisition metadata for local spatial matching.

The catalog includes the core 2021/2022 program years and additional 2022/2023
coverage. Candidate matching selects the required program year for each state."""

import argparse

import geopandas as gpd
import planetary_computer as pc
import pystac_client
import shapely.geometry

STATE_YEARS = {
    "de": ["2021", "2023"],
    "md": ["2021", "2023"],
    "ny": ["2021", "2022"],
    "va": ["2021"],
    "pa": ["2022"],
    "wv": ["2022"],
}
# DC items carry no naip:state; grab by bbox
DC_BBOX = [-77.12, 38.79, -76.90, 39.00]


def main(args: argparse.Namespace) -> None:
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    records = []

    def append_items(items, tag):
        count = 0
        for item in items:
            properties = item.properties
            epsg = (
                properties.get("proj:epsg")
                or str(properties.get("proj:code", "")).split(":")[-1]
            )
            records.append(
                dict(
                    id=item.id,
                    state=properties.get("naip:state"),
                    year=properties.get("naip:year"),
                    datetime=properties["datetime"],
                    gsd=properties.get("gsd"),
                    epsg=int(epsg),
                    href=item.assets["image"].href.split("?")[0],
                    geometry=shapely.geometry.shape(item.geometry),
                )
            )
            count += 1
        print(tag, count, flush=True)

    pairs = [(state, year) for state, years in STATE_YEARS.items() for year in years]
    if args.pairs:
        pairs = [tuple(x.split(":")) for x in args.pairs.split(",")]
    for state, year in pairs:
        append_items(
            catalog.search(
                collections=["naip"],
                query={"naip:state": {"eq": state}, "naip:year": {"eq": year}},
            ).items(),
            f"{state} {year}",
        )
    if not args.pairs:
        append_items(
            catalog.search(
                collections=["naip"], bbox=DC_BBOX, datetime="2021-01-01/2023-12-31"
            ).items(),
            "dc bbox 2021-2023",
        )

    catalog_rows = gpd.GeoDataFrame(
        records, geometry="geometry", crs="EPSG:4326"
    ).drop_duplicates("id")
    catalog_rows["datetime"] = catalog_rows["datetime"].astype(str)
    catalog_rows["month"] = catalog_rows["datetime"].str[5:7].astype(int)
    catalog_rows["leaf_state"] = catalog_rows["month"].map(
        lambda month: (
            "leaf_on"
            if 5 <= month <= 9
            else "transitional"
            if month == 10
            else "leaf_off"
        )
    )
    catalog_rows.to_parquet(args.out)
    print(len(catalog_rows), "items ->", args.out)
    print(catalog_rows.groupby(["state", "year", "leaf_state"]).size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--pairs",
        default="",
        help="override: comma-separated state:year pairs, e.g. md:2018",
    )
    main(parser.parse_args())
