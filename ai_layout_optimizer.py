import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
import shapely.affinity as affinity
import pandas as pd
import os

# YOUR ACTUAL CONSTRAINT RULES
CONSTRAINT_RULES = {
    'parcel_boundary_setback_ft': 50,
    'riverine_setback_ft': 50,
    'primary_road_gap_ft': 30,
    'secondary_road_gap_ft': 8,
    'average_gap_ft': 19,
    'wetlands_buffer_ft': 100,
    'floodplain_buffer_ft': 25
}

def get_utm_crs_for_geometry(geometry):
    """Get appropriate UTM CRS for a geometry"""
    if hasattr(geometry, 'centroid'):
        centroid = geometry.centroid
        lon, lat = centroid.x, centroid.y
    else:
        minx, miny, maxx, maxy = geometry.bounds
        lon, lat = (minx + maxx) / 2, (miny + maxy) / 2
    
    utm_zone = int((lon + 180) / 6) + 1
    
    if lat >= 0:
        epsg_code = 32600 + utm_zone
    else:
        epsg_code = 32700 + utm_zone
    
    return f"EPSG:{epsg_code}"

def compute_nominal_capacity_ac(buildable_area, mod_width, rack_length, mw_dc_per_tracker, dcac_ratio=1.2, substation_area=20, availability=0.95):
    """
    YOUR EXACT CAPACITY CALCULATION - Copied from your code
    """
    if buildable_area <= 0:
        return 0.0
    adjusted_area = max(buildable_area - substation_area, 0) * availability

    gcr = dcac_ratio / 4  # consistent with Nominal Capacity Mode
    pitch = mod_width / gcr
    area_per_tracker_ft2 = rack_length * pitch
    area_per_tracker_acres = area_per_tracker_ft2 / 43560
    tracker_count = adjusted_area / area_per_tracker_acres
    dc_capacity = tracker_count * mw_dc_per_tracker
    ac_capacity = dc_capacity / dcac_ratio

    return round(ac_capacity, 4)

def load_module_specs_from_excel():
    MODULE_FILE = "data/Module_and_CapEx_250409.xlsx"
    if not os.path.exists(MODULE_FILE):
        raise FileNotFoundError(f"❌ Required module spec file not found: {MODULE_FILE}")

    df = pd.read_excel(MODULE_FILE, sheet_name="Module Database")
    cod_column = [col for col in df.columns if "cod" in col.lower()][0]
    df[cod_column] = df[cod_column].astype(int)
    latest_row = df[df[cod_column] == df[cod_column].max()].iloc[0]

    return {
        'mod_width': latest_row["Mod Width (ft):"],
        'rack_length': latest_row["Total Rack Length (ft):"],
        'mw_dc_per_tracker': latest_row["MW DC per Tracker:"],
        'manufacturer': latest_row["Module Manufacturer:"],
        'bin_class': latest_row["Bin Class:"],
        'table_length': latest_row["Table Length (ft):"]
    }

def compute_nominal_capacity_by_parcel(gdf, dcac_ratio=1.2, substation_area=20, availability=0.95):
    """
    YOUR EXACT FUNCTION - Copied from your code
    """
    module_specs = load_module_specs_from_excel()
    
    mod_width = module_specs['mod_width']
    rack_length = module_specs['rack_length']
    mw_dc_per_tracker = module_specs['mw_dc_per_tracker']

    gdf = gdf.copy()
    gdf["nominal_mwac"] = gdf["buildable_area"].apply(
        lambda area: compute_nominal_capacity_ac(area, mod_width, rack_length, mw_dc_per_tracker, dcac_ratio, substation_area, availability)
    )
    return gdf

def apply_constraint_setbacks(geometry_utm, parcel_row, constraint_columns):
    """Apply YOUR constraint rules with proper setbacks"""
    buildable = geometry_utm.buffer(-CONSTRAINT_RULES['parcel_boundary_setback_ft'])
    
    for constraint_col in constraint_columns:
        if constraint_col in parcel_row and parcel_row[constraint_col] > 0:
            
            # Apply specific setback rules based on constraint type
            if 'wetland' in constraint_col.lower():
                setback_ft = CONSTRAINT_RULES['wetlands_buffer_ft']
            elif 'floodplain' in constraint_col.lower():
                setback_ft = CONSTRAINT_RULES['floodplain_buffer_ft']
            elif 'river' in constraint_col.lower():
                setback_ft = CONSTRAINT_RULES['riverine_setback_ft']
            elif 'road' in constraint_col.lower():
                if 'primary' in constraint_col.lower():
                    setback_ft = CONSTRAINT_RULES['primary_road_gap_ft']
                else:
                    setback_ft = CONSTRAINT_RULES['secondary_road_gap_ft']
            else:
                setback_ft = 25  # Default setback
            
            buildable = buildable.buffer(-setback_ft)
    
    return buildable

def optimize_ai_parameters(parcel_gdf, base_params, scenario_type="Balanced"):
    """AI parameter optimization using YOUR module specs"""
    module_specs = load_module_specs_from_excel()
    
    # Use YOUR module specs
    optimized_params = base_params.copy()
    optimized_params['tracker_length'] = module_specs['rack_length'] * 0.3048  # Convert to meters
    optimized_params['tracker_width'] = module_specs['mod_width'] * 0.3048   # Convert to meters
    
    if scenario_type == "Max Capacity":
        optimized_params['dcac_ratio'] = 1.1
        optimized_params['road_gap_strategy'] = 'secondary'
    elif scenario_type == "Max Efficiency":
        optimized_params['dcac_ratio'] = 1.3
        optimized_params['road_gap_strategy'] = 'primary'
    elif scenario_type == "Constraint Optimized":
        optimized_params['dcac_ratio'] = 1.25
        optimized_params['road_gap_strategy'] = 'adaptive'
    else:  # Balanced
        optimized_params['dcac_ratio'] = 1.2
        optimized_params['road_gap_strategy'] = 'average'
    
    # Calculate optimal spacing using YOUR formulas
    gcr = optimized_params['dcac_ratio'] / 4  # Your exact formula
    pitch_ft = module_specs['mod_width'] / gcr
    optimized_params['pitch_m'] = pitch_ft * 0.3048  # Convert to meters
    optimized_params['gcr'] = gcr
    optimized_params['module_specs'] = module_specs
    
    return optimized_params

def generate_ai_scenarios(parcel_gdf, base_params, constraint_cols):
    """Generate scenarios using YOUR capacity calculation methods"""
    scenarios = {}
    scenario_types = ["Max Capacity", "Max Efficiency", "Balanced", "Constraint Optimized"]
    
    for scenario_type in scenario_types:
        # Get optimized parameters
        ai_params = optimize_ai_parameters(parcel_gdf, base_params, scenario_type)
        
        # Generate layout
        layout_gdf = generate_ai_layout(parcel_gdf, ai_params, constraint_cols)
        
        # Calculate metrics using YOUR methods
        metrics = calculate_realistic_metrics(layout_gdf, parcel_gdf, ai_params)
        
        scenarios[scenario_type] = {
            'layout_gdf': layout_gdf,
            'params': ai_params,
            'metrics': metrics
        }
    
    return scenarios

def generate_ai_layout(parcel_gdf, ai_params, exclude_columns=None):
    """
    Generate layout with tracker rows — DEBUG VERSION.
    Prints info about buildable area, rows added, etc.
    """
    import geopandas as gpd
    from shapely.geometry import LineString
    import shapely.affinity as affinity

    layout_rows = []
    exclude_columns = exclude_columns or []

    utm_crs = None
    for _, row in parcel_gdf.iterrows():
        try:
            utm_crs = get_utm_crs_for_geometry(row.geometry)
            break
        except Exception as e:
            print(f"[ERROR] Failed to get UTM CRS: {e}")
            continue

    if utm_crs is None:
        utm_crs = "EPSG:32633"
        print("[INFO] Defaulting to UTM EPSG:32633")

    for idx, row in parcel_gdf.iterrows():
        try:
            geom = row.geometry

            # Convert to UTM
            geom_series = gpd.GeoSeries([geom], crs=parcel_gdf.crs)
            geom_utm = geom_series.to_crs(utm_crs).iloc[0]

            # Apply setbacks & constraints
            buildable = apply_constraint_setbacks(geom_utm, row, exclude_columns)

            if buildable.is_empty or not buildable.is_valid:
                print(f"[WARN] Parcel {row['parcel_name']} — buildable area is empty or invalid after setbacks.")
                continue

            print(f"[INFO] Parcel {row['parcel_name']} — buildable area OK. Bounds: {buildable.bounds}")

            minx, miny, maxx, maxy = buildable.bounds
            pitch_m = ai_params['pitch_m']
            angle = 0 if ai_params['orientation'] == 'North-South' else 90

            y = miny
            row_count = 0
            tracker_rows_added = 0

            while y < maxy:
                try:
                    line = LineString([(minx, y), (maxx, y)])
                    line_rot = affinity.rotate(line, angle, origin='center')
                    clipped = line_rot.intersection(buildable)

                    if (not clipped.is_empty and 
                        hasattr(clipped, 'length') and 
                        clipped.length > ai_params['tracker_length'] * 0.5):

                        module_specs = ai_params['module_specs']
                        tracker_length_m = clipped.length
                        full_tables = int(tracker_length_m / (module_specs['rack_length'] * 0.3048))
                        modules_on_tracker = full_tables * 4 * 27  # Assuming fixed here

                        layout_rows.append({
                            "geometry": clipped,
                            "parcel_name": row.parcel_name,
                            "tracker_length": tracker_length_m,
                            "row_number": row_count,
                            "full_tables": full_tables,
                            "modules_count": modules_on_tracker,
                            "tracker_type": f"Table_{full_tables}x4strings"
                        })
                        tracker_rows_added += 1
                        print(f"[DEBUG] Added tracker row {row_count} to parcel {row['parcel_name']} — length: {tracker_length_m:.2f} m, tables: {full_tables}, modules: {modules_on_tracker}")
                    else:
                        print(f"[TRACE] Skipping row {row_count} at y={y:.2f} — no valid intersection or too short.")

                except Exception as e:
                    print(f"[ERROR] Exception while creating tracker row {row_count} for parcel {row['parcel_name']}: {e}")

                y += pitch_m
                row_count += 1

            if tracker_rows_added == 0:
                print(f"[WARN] No tracker rows generated for parcel {row['parcel_name']}")

        except Exception as e:
            print(f"[ERROR] Skipping parcel {row.get('parcel_name', 'Unknown')} due to exception: {e}")
            continue

    if not layout_rows:
        print("[ERROR] No tracker rows were generated for any parcel.")
        return gpd.GeoDataFrame(columns=['geometry', 'parcel_name'], crs=parcel_gdf.crs)

    try:
        layout_gdf = gpd.GeoDataFrame(layout_rows, crs=utm_crs).to_crs(parcel_gdf.crs)
        print(f"[SUCCESS] Generated layout with {len(layout_gdf)} tracker rows.")
        return layout_gdf
    except Exception as e:
        print(f"[ERROR] Failed to create layout GeoDataFrame: {e}")
        return gpd.GeoDataFrame(columns=['geometry', 'parcel_name'], crs=parcel_gdf.crs)


def calculate_realistic_metrics(layout_gdf, parcel_gdf, ai_params):
    """Calculate metrics using YOUR capacity formulas"""
    if layout_gdf.empty:
        return {
            'total_tracker_rows': 0,
            'total_tables': 0,
            'total_modules': 0,
            'estimated_dc_capacity_mw': 0,
            'estimated_ac_capacity_mw': 0,
            'capacity_density_mw_acre': 0,
            'land_utilization_pct': 0,
            'gcr': ai_params.get('gcr', 0),
            'dcac_ratio': ai_params.get('dcac_ratio', 1.2),
            'nominal_capacity_check': 0
        }
    
    try:
        module_specs = ai_params['module_specs']
        
        # YOUR calculations
        total_rows = len(layout_gdf)
        total_tables = layout_gdf['full_tables'].sum() if 'full_tables' in layout_gdf.columns else 0
        total_modules = layout_gdf['modules_count'].sum() if 'modules_count' in layout_gdf.columns else 0
        
        # DC capacity using YOUR module specs
        dc_capacity_mw = total_tables * module_specs['mw_dc_per_tracker']
        
        # AC capacity using YOUR formula
        dcac_ratio = ai_params.get('dcac_ratio', 1.2)
        ac_capacity_mw = dc_capacity_mw / dcac_ratio
        
        # Compare with YOUR nominal capacity calculation
        total_buildable = parcel_gdf['buildable_area'].sum() if 'buildable_area' in parcel_gdf.columns else parcel_gdf['total_acres'].sum()
        nominal_capacity = compute_nominal_capacity_ac(
            total_buildable, 
            module_specs['mod_width'], 
            module_specs['rack_length'], 
            module_specs['mw_dc_per_tracker'], 
            dcac_ratio
        )
        
        # Land utilization
        total_area_acres = parcel_gdf['total_acres'].sum()
        capacity_density = ac_capacity_mw / total_area_acres if total_area_acres > 0 else 0
        
        # Module area calculation
        module_area_ft2 = module_specs['mod_width'] * 6.9  # Approximate module length
        total_module_area_acres = (total_modules * module_area_ft2) / 43560
        land_utilization = (total_module_area_acres / total_area_acres) * 100 if total_area_acres > 0 else 0
        
        return {
            'total_tracker_rows': total_rows,
            'total_tables': int(total_tables),
            'total_modules': int(total_modules),
            'estimated_dc_capacity_mw': round(dc_capacity_mw, 1),
            'estimated_ac_capacity_mw': round(ac_capacity_mw, 1),
            'capacity_density_mw_acre': round(capacity_density, 3),
            'land_utilization_pct': round(land_utilization, 1),
            'gcr': round(ai_params.get('gcr', 0), 3),
            'dcac_ratio': dcac_ratio,
            'nominal_capacity_check': round(nominal_capacity, 1),
            'module_specs': module_specs['manufacturer'],
            'constraint_rules_applied': True
        }
        
    except Exception as e:
        print(f"Warning calculating metrics: {e}")
        return {
            'total_tracker_rows': len(layout_gdf) if not layout_gdf.empty else 0,
            'total_tables': 0,
            'total_modules': 0,
            'estimated_dc_capacity_mw': 0,
            'estimated_ac_capacity_mw': 0,
            'capacity_density_mw_acre': 0,
            'land_utilization_pct': 0,
            'gcr': ai_params.get('gcr', 0),
            'dcac_ratio': ai_params.get('dcac_ratio', 1.2),
            'nominal_capacity_check': 0
        }

def get_constraint_rules():
    """Return the constraint rules for display"""
    return CONSTRAINT_RULES

def get_module_specifications():
    module_specs = load_module_specs_from_excel()
    return {
        'Module Details': {
            'Module Width (ft)': module_specs['mod_width'],
            'Rack Length (ft)': module_specs['rack_length'],
            'MW DC per Tracker': module_specs['mw_dc_per_tracker'],
            'Manufacturer': module_specs['manufacturer'],
            'Bin Class': module_specs['bin_class'],
            'Table Length (ft)': module_specs['table_length']
        },
        'Constraint Rules': get_constraint_rules()
    }



# Aliases for compatibility
calculate_layout_metrics = calculate_realistic_metrics 