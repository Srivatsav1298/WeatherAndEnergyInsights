# map_pages.py
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import numpy as np

GEOJSON_PATH = os.path.join("..", "data", "elspot_areas.geojson")
CENTROIDS_CSV = os.path.join("data", "price_areas_cities.csv")

def page_price_area_map_selectable(df_elhub):
    """
    Selectable price-area map. Stores selection in st.session_state:
      - selected_price_area (string)
      - selected_coord (tuple lat, lon) or None
    """
    st.header("Price Area Map — selectable")
    if df_elhub is None or df_elhub.empty:
        st.error("Production data not loaded. Please ensure Mongo is accessible.")
        return

    # ensure datetime
    if "startTime" in df_elhub.columns:
        df_elhub["startTime"] = pd.to_datetime(df_elhub["startTime"])

    st.markdown("Choose date window and production group to paint map by mean production.")

    min_dt = df_elhub["startTime"].min().date()
    max_dt = df_elhub["startTime"].max().date()
    st.markdown(f"Data range: **{min_dt}** → **{max_dt}**")
    start_dt = st.date_input("Start date", min_dt, min_value=min_dt, max_value=max_dt)
    end_dt = st.date_input("End date", max_dt, min_value=min_dt, max_value=max_dt)
    if start_dt > end_dt:
        st.error("Start date must be <= End date.")
        return

    pgroups = sorted(df_elhub["productionGroup"].dropna().unique())
    chosen_groups = st.multiselect("Production groups", pgroups, default=pgroups[:1])

    mask = (df_elhub["startTime"].dt.date >= start_dt) & (df_elhub["startTime"].dt.date <= end_dt)
    df_sel = df_elhub.loc[mask & df_elhub["productionGroup"].isin(chosen_groups)].copy()
    if df_sel.empty:
        st.warning("No production data for selection.")
        return

    agg = df_sel.groupby("priceArea")["quantityKwh"].mean().reset_index().rename(columns={"quantityKwh": "mean_kwh"})
    st.dataframe(agg)

    # Try geojson
    geojson_obj = None
    featureidkey = None
    geojson_loaded = False
    if os.path.exists(GEOJSON_PATH):
        try:
            with open(GEOJSON_PATH, "r", encoding="utf-8") as fh:
                geojson_obj = json.load(fh)
            # detect property key that contains 'NO' code
            if "features" in geojson_obj and len(geojson_obj["features"])>0:
                props = geojson_obj["features"][0].get("properties", {})
                candidate = None
                for k in props.keys():
                    sample_vals = {feat.get("properties",{}).get(k) for feat in geojson_obj["features"][:30]}
                    if any(isinstance(x,str) and x.strip().upper().startswith("NO") for x in sample_vals if x):
                        candidate = k
                        break
                if candidate:
                    featureidkey = f"properties.{candidate}"
                    geojson_loaded = True
        except Exception as e:
            st.warning(f"Could not read GeoJSON: {e}")

    # Construct options list (agg priceArea plus ones in geojson)
    options = sorted(list(agg["priceArea"].dropna().unique()))
    if geojson_loaded:
        # append features found in geojson
        vals = []
        for feat in geojson_obj["features"]:
            p = feat.get("properties",{})
            key = featureidkey.split(".",1)[1]
            v = p.get(key)
            if v is not None:
                vals.append(str(v))
        for v in sorted(set(vals)):
            if v not in options:
                options.append(v)

    selected_pa = st.selectbox("Choose price area", options, index=0)
    # derive centroid either from geojson or csv
    centroid = None
    if geojson_loaded:
        key = featureidkey.split(".",1)[1]
        for feat in geojson_obj["features"]:
            if str(feat.get("properties", {}).get(key)) == str(selected_pa):
                # compute coarse centroid
                geom = feat.get("geometry", {})
                coords = []
                def collect(g):
                    typ = g.get("type")
                    c = g.get("coordinates")
                    if typ == "Polygon":
                        for ring in c: coords.extend(ring)
                    elif typ == "MultiPolygon":
                        for poly in c:
                            for ring in poly: coords.extend(ring)
                collect(geom)
                if coords:
                    lons = [pt[0] for pt in coords]
                    lats = [pt[1] for pt in coords]
                    centroid = (float(np.mean(lats)), float(np.mean(lons)))
                    break

    if centroid is None and os.path.exists(CENTROIDS_CSV):
        centers = pd.read_csv(CENTROIDS_CSV)
        centers = centers.rename(columns={"price_area":"priceArea","longitude":"lon","latitude":"lat"})
        row = centers[centers["priceArea"]==selected_pa]
        if not row.empty:
            centroid = (float(row["lat"].values[0]), float(row["lon"].values[0]))

    # store selection
    st.session_state["selected_price_area"] = selected_pa
    st.session_state["selected_coord"] = centroid

    st.write(f"Selected: **{selected_pa}** — centroid: {centroid}")

    # Plot choropleth if possible
    if geojson_loaded:
        try:
            fig = px.choropleth_mapbox(
                agg,
                geojson=geojson_obj,
                locations="priceArea",
                featureidkey=featureidkey,
                color="mean_kwh",
                color_continuous_scale="Viridis",
                mapbox_style="carto-positron",
                center={"lat":63.5,"lon":10.0},
                zoom=4.2,
                opacity=0.5
            )
            # highlight centroid
            if centroid is not None:
                fig.add_trace(px.scatter_mapbox(
                    pd.DataFrame([{"lat":centroid[0],"lon":centroid[1],"label":selected_pa}]),
                    lat="lat", lon="lon", hover_name="label"
                ).data[0])
            fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render choropleth: {e}")
    else:
        # centroid fallback
        if os.path.exists(CENTROIDS_CSV):
            centers = pd.read_csv(CENTROIDS_CSV).rename(columns={"price_area":"priceArea","longitude":"lon","latitude":"lat"})
            merged = centers.merge(agg, on="priceArea", how="left").fillna(0)
            fig = px.scatter_mapbox(
                merged, lat="lat", lon="lon", size="mean_kwh", color="mean_kwh",
                hover_name="priceArea", mapbox_style="carto-positron", zoom=4.2
            )
            # add selected marker
            if centroid:
                fig.add_trace(px.scatter_mapbox(
                    pd.DataFrame([{"lat":centroid[0],"lon":centroid[1],"label":selected_pa}]),
                    lat="lat", lon="lon", hover_name="label"
                ).data[0])
            fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No GeoJSON and no centroid CSV available.")

