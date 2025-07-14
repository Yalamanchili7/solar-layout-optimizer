import geopandas as gpd
from shapely.geometry import LineString
import shapely.affinity as affinity

def get_utm_crs_for_geometry(geometry):
    """Get appropriate UTM CRS for a geometry"""
    # Get the centroid to determine UTM zone
    if hasattr(geometry, 'centroid'):
        centroid = geometry.centroid
        lon, lat = centroid.x, centroid.y
    else:
        # Fallback: get bounds center
        minx, miny, maxx, maxy = geometry.bounds
        lon, lat = (minx + maxx) / 2, (miny + maxy) / 2
    
    # Calculate UTM zone
    utm_zone = int((lon + 180) / 6) + 1
    
    # Determine hemisphere
    if lat >= 0:
        # Northern hemisphere
        epsg_code = 32600 + utm_zone
    else:
        # Southern hemisphere  
        epsg_code = 32700 + utm_zone
    
    return f"EPSG:{epsg_code}"

def optimize_layout(parcel_gdf, setback, tracker_length, tracker_width, gcr, orientation, exclude_columns=None):
    """
    Traditional layout optimization with fixed geometry handling
    """
    layout_rows = []
    exclude_columns = exclude_columns or []

    # Get UTM CRS from the first valid geometry
    utm_crs = None
    for _, row in parcel_gdf.iterrows():
        try:
            utm_crs = get_utm_crs_for_geometry(row.geometry)
            break
        except:
            continue
    
    if utm_crs is None:
        # Fallback to a default UTM zone
        utm_crs = "EPSG:32633"  # UTM Zone 33N

    for idx, row in parcel_gdf.iterrows():
        geom = row.geometry

        try:
            # Convert to UTM projection for accurate distance calculations
            geom_series = gpd.GeoSeries([geom], crs=parcel_gdf.crs)
            geom_utm = geom_series.to_crs(utm_crs).iloc[0]

            # Apply setback
            buildable = geom_utm.buffer(-setback)

            # Apply constraint areas
            for col in exclude_columns:
                if col in row and row[col] > 0:
                    # Simplified constraint removal - reduce buildable area
                    constraint_ratio = min(row[col] / (row.get('total_acres', 100) * 4047), 0.3)
                    buildable = buildable.buffer(-constraint_ratio * 50)

            if buildable.is_empty or not buildable.is_valid:
                continue

            minx, miny, maxx, maxy = buildable.bounds
            pitch = tracker_width / gcr
            angle = 0 if orientation == "North-South" else 90

            y = miny
            while y < maxy:
                try:
                    line = LineString([(minx, y), (maxx, y)])
                    line_rot = affinity.rotate(line, angle, origin='center')
                    clipped = line_rot.intersection(buildable)
                    
                    if not clipped.is_empty and hasattr(clipped, 'length'):
                        layout_rows.append({
                            "geometry": clipped,
                            "parcel_name": row.parcel_name
                        })
                except Exception as e:
                    # Skip problematic rows
                    print(f"Warning: Skipping row due to error: {e}")
                    pass
                    
                y += pitch
                
        except Exception as e:
            print(f"Warning: Skipping parcel {row.get('parcel_name', 'Unknown')} due to error: {e}")
            continue

    if not layout_rows:
        return gpd.GeoDataFrame(columns=['geometry', 'parcel_name'], crs=parcel_gdf.crs)

    try:
        layout_gdf = gpd.GeoDataFrame(layout_rows, crs=utm_crs).to_crs(parcel_gdf.crs)
        return layout_gdf
    except Exception as e:
        print(f"Warning: Error creating layout GeoDataFrame: {e}")
        return gpd.GeoDataFrame(columns=['geometry', 'parcel_name'], crs=parcel_gdf.crs)