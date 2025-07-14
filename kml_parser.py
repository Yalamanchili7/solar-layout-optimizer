import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import reverse_geocode
import re
import streamlit as st
import os

# Load county shapefile using relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
county_fp = os.path.join(BASE_DIR, "data/tl_2023_us_county/tl_2023_us_county.shp")

def is_parcel_name(name): 
    """Check if name represents a parcel"""
    return name and " - " in name and re.match(r"^[0-9.]+$", name.split(" - ")[-1].strip())

def extract_owner(name):
    """Extract owner from parcel name"""
    return name.split(" - ")[0].strip() if " - " in name else name.strip()

def extract_geometry(pm):
    """Extract geometry from placemark"""
    coords_elem = pm.find(".//coordinates")
    if coords_elem is None or coords_elem.text is None:
        return None
    coord_pairs = coords_elem.text.strip().split()
    points = [(float(pt.split(",")[0]), float(pt.split(",")[1])) for pt in coord_pairs if len(pt.split(",")) >= 2]
    return Polygon(points).buffer(0) if len(points) >= 3 else None

def normalize_name(name):
    """Normalize name for comparison"""
    return name.lower().strip().replace("–", "-").replace("—", "-")

def classify_overlay_type(name, description):
    """Classify overlay based on name and description"""
    name_lower = name.lower()
    desc_lower = description.lower() if description else ""
    
    # Classification rules based on common Anderson KML patterns
    if any(keyword in name_lower for keyword in ['wetland', 'wet']):
        return 'wetlands_area_m²'
    elif any(keyword in name_lower for keyword in ['flood', 'fema']):
        return 'floodplain_area_m²'
    elif any(keyword in name_lower for keyword in ['slope', 'steep']):
        if '5' in name_lower and '10' in name_lower:
            return 'slope_5_10pct_area_m²'
        elif '10' in name_lower and '15' in name_lower:
            return 'slope_10_15pct_area_m²'
        elif '15' in name_lower:
            return 'slope_15pct_area_m²'
        else:
            return 'slope_area_m²'
    elif any(keyword in name_lower for keyword in ['road', 'highway', 'street']):
        return 'roads_length_m'
    elif any(keyword in name_lower for keyword in ['building', 'structure']):
        return 'buildings_area_m²'
    elif any(keyword in name_lower for keyword in ['bedrock', 'rock']):
        return 'bedrock_area_m²'
    elif any(keyword in name_lower for keyword in ['river', 'stream', 'creek']):
        return 'riverine_area_m²'
    elif any(keyword in name_lower for keyword in ['setback', 'buffer']):
        return 'setback_area_m²'
    else:
        # Generic constraint
        return f"{name_lower.replace(' ', '_')}_area_m²"

def find_overlapping_parcels(overlay_geom, parcels_gdf):
    """Find which parcels this overlay intersects with"""
    overlapping_parcels = []
    
    for idx, parcel_row in parcels_gdf.iterrows():
        try:
            if overlay_geom.intersects(parcel_row.geometry):
                # Calculate intersection area
                intersection = overlay_geom.intersection(parcel_row.geometry)
                if hasattr(intersection, 'area') and intersection.area > 0:
                    # Convert to square meters (approximate)
                    area_m2 = intersection.area * 111320 * 111320  # Rough conversion from degrees
                    overlapping_parcels.append({
                        'parcel_name': parcel_row.parcel_name,
                        'parcel_idx': idx,
                        'intersection_area_m2': area_m2
                    })
        except Exception as e:
            print(f"Warning: Error checking intersection for {parcel_row.get('parcel_name', 'Unknown')}: {e}")
            continue
    
    return overlapping_parcels

def parse_anderson_kml(kml_file_path):
    """Enhanced KML parser that extracts and matches overlays to parcels"""
    
    with open(kml_file_path, "r", encoding="utf-8") as f:
        raw_kml = f.read()

    root = ET.fromstring(raw_kml)
    parcels = []
    all_overlays = []
    buildable_by_parcel = {}

    # First pass: Extract all parcels and overlays
    def parse_folder(folder, current_parcel=None):
        for child in folder:
            if child.tag.endswith("Folder"):
                name_elem = child.find("name")
                name = name_elem.text.strip() if name_elem is not None else None
                parse_folder(child, current_parcel=name if is_parcel_name(name) else current_parcel)
                
            elif child.tag.endswith("Placemark"):
                name_elem = child.find("name")
                desc_elem = child.find("description")
                name = name_elem.text.strip() if name_elem is not None else "Unnamed"
                metadata = desc_elem.text if desc_elem is not None else ""
                geometry = extract_geometry(child)

                if geometry is None or not geometry.is_valid:
                    continue

                name_clean = normalize_name(name)
                metadata_clean = metadata.lower().strip()

                # Check if this is a parcel
                if is_parcel_name(name_clean):
                    parcels.append({
                        "parcel_name": name, 
                        "geometry": geometry, 
                        "owner": extract_owner(name)
                    })
                else:
                    # This is an overlay/constraint
                    area_match = re.search(r"area[:\s]*([\d.]+)", metadata_clean)
                    area_val = float(area_match.group(1)) if area_match else 0.0

                    # Classify the overlay type
                    overlay_type = classify_overlay_type(name, metadata)
                    
                    overlay = {
                        "name": name,
                        "geometry": geometry, 
                        "area": area_val,
                        "type": overlay_type,
                        "original_name": name,
                        "description": metadata,
                        "parent_parcel": current_parcel
                    }
                    
                    all_overlays.append(overlay)

                # Check for buildable area info
                buildable_match = re.search(r"buildable \(acre\):\s*([\d.]+)", metadata_clean)
                pct_match = re.search(r"buildable \(%\):\s*([\d.]+)", metadata_clean)
                if buildable_match or pct_match:
                    if current_parcel:
                        normalized = normalize_name(current_parcel)
                        buildable_by_parcel[normalized] = {
                            "area": float(buildable_match.group(1)) if buildable_match else 0.0,
                            "pct": float(pct_match.group(1)) if pct_match else 0.0
                        }

    parse_folder(root)

    if not parcels:
        raise ValueError("No valid parcels found in KML.")

    # Create initial parcel GeoDataFrame
    parcel_gdf = gpd.GeoDataFrame(parcels, crs="EPSG:4326")
    parcel_gdf["total_acres"] = parcel_gdf["parcel_name"].apply(lambda x: float(x.split(" - ")[-1]))
    
    # Add state information
    try:
        parcel_gdf["centroid"] = parcel_gdf.geometry.centroid
        parcel_gdf["state"] = parcel_gdf["centroid"].apply(
            lambda pt: reverse_geocode.search([(pt.y, pt.x)])[0]["state"]
        )
        parcel_gdf.drop(columns=["centroid"], inplace=True)
    except:
        parcel_gdf["state"] = "Unknown"

    # Initialize buildable area columns
    parcel_gdf["buildable_area"] = 0.0
    parcel_gdf["buildable_pct"] = 0.0

    # Get all unique constraint types from overlays
    all_constraint_types = set(overlay["type"] for overlay in all_overlays)
    
    # Initialize constraint columns
    for constraint_type in all_constraint_types:
        parcel_gdf[constraint_type] = 0.0

    print(f"Found {len(all_overlays)} overlays with {len(all_constraint_types)} unique constraint types")
    
    # Second pass: Match overlays to parcels and calculate intersections
    overlay_matches = []
    
    for overlay in all_overlays:
        try:
            # Find which parcels this overlay intersects
            overlapping_parcels = find_overlapping_parcels(overlay["geometry"], parcel_gdf)
            
            if overlapping_parcels:
                for match in overlapping_parcels:
                    parcel_idx = match["parcel_idx"]
                    intersection_area = match["intersection_area_m2"]
                    constraint_type = overlay["type"]
                    
                    # Add constraint area to the parcel
                    current_value = parcel_gdf.at[parcel_idx, constraint_type]
                    parcel_gdf.at[parcel_idx, constraint_type] = current_value + intersection_area
                    
                    overlay_matches.append({
                        "overlay_name": overlay["name"],
                        "parcel_name": match["parcel_name"],
                        "constraint_type": constraint_type,
                        "area_m2": intersection_area
                    })
            else:
                # No intersection found - might be a general constraint
                print(f"Warning: Overlay '{overlay['name']}' doesn't intersect with any parcels")
                
        except Exception as e:
            print(f"Warning: Error processing overlay '{overlay.get('name', 'Unknown')}': {e}")
            continue

    # Fill buildable area information
    for idx, row in parcel_gdf.iterrows():
        pname = normalize_name(row["parcel_name"])
        
        if pname in buildable_by_parcel:
            parcel_gdf.at[idx, "buildable_area"] = buildable_by_parcel[pname].get("area", 0.0)
            parcel_gdf.at[idx, "buildable_pct"] = buildable_by_parcel[pname].get("pct", 0.0)
        else:
            # Estimate buildable area if not provided
            total_constraint_area = sum(
                parcel_gdf.at[idx, col] for col in all_constraint_types 
                if col in parcel_gdf.columns
            )
            total_area_m2 = row["total_acres"] * 4047  # Convert to m²
            estimated_buildable_m2 = max(0, total_area_m2 - total_constraint_area)
            estimated_buildable_acres = estimated_buildable_m2 / 4047
            
            parcel_gdf.at[idx, "buildable_area"] = estimated_buildable_acres
            parcel_gdf.at[idx, "buildable_pct"] = (estimated_buildable_acres / row["total_acres"]) * 100

    # Create overlay GeoDataFrame for visualization
    if all_overlays:
        overlay_gdf = gpd.GeoDataFrame(all_overlays, crs="EPSG:4326")
        st.session_state["overlay_metrics"] = overlay_gdf
    else:
        st.session_state["overlay_metrics"] = gpd.GeoDataFrame()

    # Store overlay matching results
    if overlay_matches:
        st.session_state["overlay_matches"] = overlay_matches
        print(f"Successfully matched {len(overlay_matches)} overlay-parcel intersections")

    print(f"Final constraint columns: {sorted(all_constraint_types)}")
    
    return parcel_gdf