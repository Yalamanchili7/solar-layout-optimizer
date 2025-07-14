import folium
from folium.plugins import MeasureControl
from shapely.geometry import mapping
from branca.colormap import linear
import numpy as np

def render_parcel_map(
    parcel_gdf,
    layout_gdf=None,
    overlay_gdf=None,
    show_modules=True
):
    """
    Enhanced map renderer that shows individual modules on trackers
    """
    
    # Color schemes
    CONSTRAINT_COLOR_MAP = {
        "wetlands_area_m²": "#9ecae1",
        "bedrock_area_m²": "#bcbddc",
        "roads_length_m": "#969696",
        "setback_area_m²": "#fdd49e",
        "floodplain_area_m²": "#ccece6",
        "buffer_zone": "#fcae91",
        "slope_5_10pct_area_m²": "#ffffb2",
        "slope_10_15pct_area_m²": "#fecc5c",
        "slope_15pct_area_m²": "#fd8d3c",
        "riverine_area_m²": "#a1dab4",
        "buildings_area_m²": "#756bb1"
    }

    # Initialize map
    center_lat = parcel_gdf.geometry.centroid.y.mean()
    center_lon = parcel_gdf.geometry.centroid.x.mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        control_scale=True,
        tiles="OpenStreetMap"
    )

    # Add satellite layer
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Satellite",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        control=True
    ).add_to(m)

    # Feature groups for layer control
    parcel_group = folium.FeatureGroup(name="📍 Parcel Boundaries", show=True)
    constraint_group = folium.FeatureGroup(name="🚫 Constraints", show=True)
    tracker_group = folium.FeatureGroup(name="🔆 Tracker Rows", show=True)
    module_group = folium.FeatureGroup(name="📱 Individual Modules", show=show_modules)
    labels_group = folium.FeatureGroup(name="🏷️ Labels", show=True)
    
    # Render constraints first (background)
    if overlay_gdf is not None and not overlay_gdf.empty:
        constraint_types = overlay_gdf["type"].unique() if "type" in overlay_gdf.columns else []
        
        for constraint_type in constraint_types:
            if constraint_type in CONSTRAINT_COLOR_MAP:
                subset = overlay_gdf[overlay_gdf["type"] == constraint_type]
                color = CONSTRAINT_COLOR_MAP[constraint_type]
                
                for _, row in subset.iterrows():
                    folium.GeoJson(
                        mapping(row.geometry),
                        style_function=lambda _, c=color: {
                            "fillColor": c,
                            "color": c,
                            "weight": 1,
                            "fillOpacity": 0.4,
                            "opacity": 0.8
                        },
                        tooltip=f"Constraint: {constraint_type}",
                        popup=f"<b>{constraint_type}</b><br>Area: {row.get('area', 'N/A')} m²"
                    ).add_to(constraint_group)

    # Render parcels
    for idx, row in parcel_gdf.iterrows():
        # Enhanced tooltip with capacity info
        tooltip_data = f"""
        <div style='font-family: Arial; font-size: 12px;'>
            <b>🏞️ {row.get('parcel_name', 'Unknown')}</b><br>
            <b>Owner:</b> {row.get('owner', 'N/A')}<br>
            <b>Total Area:</b> {row.get('total_acres', 0):.1f} acres<br>
            <b>Buildable:</b> {row.get('buildable_area', 0):.1f} acres ({row.get('buildable_pct', 0):.1f}%)<br>
            <b>State:</b> {row.get('state', 'N/A')}<br>
        </div>
        """

        # Add parcel boundary
        folium.GeoJson(
            mapping(row.geometry),
            tooltip=tooltip_data,
            popup=tooltip_data,
            style_function=lambda _: {
                "fillColor": "#90EE90",
                "color": "#006400",
                "weight": 2,
                "fillOpacity": 0.1,
                "opacity": 1
            }
        ).add_to(parcel_group)

        # Add parcel label
        folium.Marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            icon=folium.DivIcon(
                html=f"""
                <div style='
                    background: rgba(255,255,255,0.9);
                    border: 2px solid #006400;
                    border-radius: 5px;
                    padding: 2px 5px;
                    font-size: 10px;
                    font-weight: bold;
                    color: #006400;
                    text-align: center;
                    white-space: nowrap;
                '>
                    {row.get('parcel_name', 'Unknown')}
                </div>
                """,
                icon_size=(None, None),
                icon_anchor=(0, 0)
            )
        ).add_to(labels_group)

    # Render tracker layout with modules
    if layout_gdf is not None and not layout_gdf.empty:
        
        # Add tracker rows (blue lines)
        for idx, row in layout_gdf.iterrows():
            tracker_length = row.get('tracker_length', 0)
            modules_count = row.get('modules_count', 0)
            full_tables = row.get('full_tables', 0)

            # Tracker row tooltip
            tracker_tooltip = f"""
            <div style='font-family: Arial; font-size: 11px;'>
                <b>🔆 Tracker Row {row.get('row_number', idx)}</b><br>
                <b>Parcel:</b> {row.get('parcel_name', 'N/A')}<br>
                <b>Length:</b> {tracker_length:.1f} m<br>
                <b>Tables:</b> {full_tables}<br>
                <b>Modules:</b> {modules_count} × Illuminate 650W<br>
                <b>DC Power:</b> {full_tables * 0.070244:.2f} MW
            </div>
            """

            folium.GeoJson(
                mapping(row.geometry),
                style_function=lambda _: {
                    "color": "#0066CC",
                    "weight": 3,
                    "opacity": 0.8
                },
                tooltip=tracker_tooltip,
                popup=tracker_tooltip
            ).add_to(tracker_group)

        # Add individual module points if requested
        if show_modules and 'modules_gdf' in layout_gdf.columns:
            for _, tracker_row in layout_gdf.iterrows():
                modules_gdf = tracker_row.get('modules_gdf')
                
                if modules_gdf is not None and not modules_gdf.empty:
                    for _, module_row in modules_gdf.iterrows():
                        # Individual module markers (small red dots)
                        folium.CircleMarker(
                            location=[module_row.geometry.y, module_row.geometry.x],
                            radius=2,
                            popup=f"""
                            <div style='font-family: Arial; font-size: 10px;'>
                                <b>📱 Module {module_row.get('module_id', 'N/A')}</b><br>
                                <b>Type:</b> Illuminate 650W<br>
                                <b>Tracker:</b> Row {module_row.get('tracker_row', 'N/A')}<br>
                                <b>Power:</b> 650W DC
                            </div>
                            """,
                            tooltip=f"Module {module_row.get('module_id', 'N/A')}",
                            color="#FF4444",
                            fillColor="#FF6666",
                            fillOpacity=0.8,
                            weight=1
                        ).add_to(module_group)

        # Add summary statistics
        total_trackers = len(layout_gdf)
        total_modules = layout_gdf['modules_count'].sum() if 'modules_count' in layout_gdf.columns else 0
        total_tables = layout_gdf['full_tables'].sum() if 'full_tables' in layout_gdf.columns else 0
        estimated_dc_mw = total_tables * 0.070244

        # Summary marker
        folium.Marker(
            location=[center_lat + 0.001, center_lon + 0.001],
            icon=folium.DivIcon(
                html=f"""
                <div style='
                    background: linear-gradient(45deg, #1e3c72, #2a5298);
                    color: white;
                    padding: 8px 12px;
                    border-radius: 8px;
                    font-size: 11px;
                    font-weight: bold;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                    border: 2px solid white;
                    max-width: 200px;
                '>
                    🔆 Layout Summary<br>
                    📊 {total_trackers} Tracker Rows<br>
                    🔧 {total_tables} Tables<br>
                    📱 {total_modules:,} Modules<br>
                    ⚡ {estimated_dc_mw:.1f} MW DC
                </div>
                """,
                icon_size=(None, None),
                icon_anchor=(0, 0)
            ),
            popup=f"""
            <div style='font-family: Arial;'>
                <h3>🔆 Layout Details</h3>
                <b>Tracker Rows:</b> {total_trackers}<br>
                <b>Total Tables:</b> {total_tables}<br>
                <b>Total Modules:</b> {total_modules:,}<br>
                <b>Module Type:</b> Illuminate 650W<br>
                <b>Estimated DC:</b> {estimated_dc_mw:.1f} MW<br>
                <b>Estimated AC:</b> {estimated_dc_mw/1.2:.1f} MW (DC/AC=1.2)
            </div>
            """
        ).add_to(tracker_group)

    # Add all feature groups to map
    parcel_group.add_to(m)
    constraint_group.add_to(m)
    tracker_group.add_to(m)
    module_group.add_to(m)
    labels_group.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False, position='topright').add_to(m)

    # Add measure control
    m.add_child(MeasureControl(
        primary_length_unit='meters',
        secondary_length_unit='feet',
        primary_area_unit='hectares',
        secondary_area_unit='acres'
    ))

    # Add custom legend
    legend_html = f"""
    <div style='
        position: fixed; 
        bottom: 50px; 
        left: 50px; 
        width: 250px; 
        height: auto;
        background-color: white; 
        border:2px solid grey; 
        z-index:9999; 
        font-size:12px;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0,0,0,0.3);
    '>
    <h4 style='margin: 0 0 10px 0;'>🗺️ Map Legend</h4>
    <div><span style='color: #006400; font-weight: bold;'>━━</span> Parcel Boundaries</div>
    <div><span style='color: #0066CC; font-weight: bold;'>━━</span> Tracker Rows</div>
    <div><span style='color: #FF4444; font-weight: bold;'>●</span> Individual Modules (650W)</div>
    <div><span style='color: #9ecae1; font-weight: bold;'>▓▓</span> Constraint Areas</div>
    <hr style='margin: 8px 0;'>
    <small><b>🔧 Specs:</b> Illuminate 650W, 4×27 modules/tracker</small>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))

    return m