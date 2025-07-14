import streamlit as st
import pandas as pd
import os
from kml_parser import parse_anderson_kml
from map_renderer import render_parcel_map
from layout_optimizer import optimize_layout
from ai_layout_optimizer import (
    generate_ai_scenarios, 
    calculate_realistic_metrics, 
    get_module_specifications,
    get_constraint_rules,
    compute_nominal_capacity_ac
)
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Solar Layout Optimizer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔆 AI Solar Layout Optimizer")
st.markdown("**Utility-Scale Optimization with Illuminate 650W Modules**")

# Sidebar Configuration
st.sidebar.header("🔧 Project Configuration")
uploaded_kml = st.sidebar.file_uploader("Upload Anderson KML", type=["kml"])

# Layout Parameters
st.sidebar.subheader("📐 Layout Parameters")
setback = st.sidebar.number_input("Additional Setback (m)", min_value=0.0, value=0.0, help="Additional setback beyond constraint rules")
tracker_length = st.sidebar.number_input("Max Tracker Length (m)", min_value=1.0, value=46.6, help="Maximum tracker length in meters")
tracker_width = st.sidebar.number_input("Tracker Width (m)", min_value=0.1, value=2.38, help="Tracker width in meters") 
orientation = st.sidebar.selectbox("Orientation", ["North-South", "East-West"])

# AI Configuration
st.sidebar.subheader("🤖 AI Optimization")
ai_enabled = st.sidebar.toggle("Enable AI Optimization", value=True)

parcel_gdf = None
layout_scenarios = {}

if uploaded_kml:
    # Process KML file
    os.makedirs("data/uploads", exist_ok=True)
    kml_path = os.path.join("data/uploads", uploaded_kml.name)
    with open(kml_path, "wb") as f:
        f.write(uploaded_kml.read())

    with st.spinner("📄 Parsing KML file..."):
        try:
            parcel_gdf = parse_anderson_kml(kml_path)
            st.success(f"✅ Successfully parsed {len(parcel_gdf)} parcels")
            
            # Project Summary
            col1, col2, col3, col4 = st.columns(4)
            total_acres = parcel_gdf['total_acres'].sum()
            buildable_acres = parcel_gdf['buildable_area'].sum() if 'buildable_area' in parcel_gdf.columns else total_acres * 0.8
            avg_buildable_pct = parcel_gdf['buildable_pct'].mean() if 'buildable_pct' in parcel_gdf.columns else 80
            
            with col1:
                st.metric("Total Area", f"{total_acres:.1f} acres")
            with col2:
                st.metric("Buildable Area", f"{buildable_acres:.1f} acres")
            with col3:
                st.metric("Buildable %", f"{avg_buildable_pct:.1f}%")
            with col4:
                # Load module specs
                module_specs = get_module_specifications()
                mod_details = module_specs['Module Details']

                mod_width = mod_details['Module Width (ft)']
                rack_length = mod_details['Rack Length (ft)']
                mw_dc_per_tracker = mod_details['MW DC per Tracker']

               # Compute nominal capacity
                nominal_capacity = compute_nominal_capacity_ac(
                   buildable_acres,
                   mod_width,
                   rack_length,
                   mw_dc_per_tracker
                )
                st.metric("Nominal Capacity", f"{nominal_capacity:.1f} MW AC")

            # Display module specifications and constraint rules
            with st.expander("🔧 Module & Constraint Specifications", expanded=False):
                module_specs = get_module_specifications()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📱 Module Details")
                    for key, value in module_specs['Module Details'].items():
                        st.write(f"**{key}:** {value}")
                
                with col2:
                    st.subheader("🚫 Constraint Rules") 
                    for key, value in module_specs['Constraint Rules'].items():
                        st.write(f"**{key}:** {value}")

            # Display constraint matching results
            if st.session_state.get("overlay_matches"):
                with st.expander("🔗 Overlay-Parcel Matching Results", expanded=False):
                    overlay_matches_df = pd.DataFrame(st.session_state["overlay_matches"])
                    st.write(f"Found {len(overlay_matches_df)} overlay-parcel intersections:")
                    st.dataframe(overlay_matches_df, use_container_width=True)
                    
                    # Show constraint summary by parcel
                    constraint_summary = overlay_matches_df.groupby(['parcel_name', 'constraint_type'])['area_m2'].sum().reset_index()
                    constraint_summary['area_acres'] = constraint_summary['area_m2'] / 4047
                    st.write("**Constraint Summary by Parcel:**")
                    st.dataframe(constraint_summary, use_container_width=True)

            # Display parcel data
            with st.expander("📊 Parcel Data Details", expanded=False):
                display_df = parcel_gdf.drop(columns=['geometry']) if 'geometry' in parcel_gdf.columns else parcel_gdf
                st.dataframe(display_df, use_container_width=True)

            # Constraint Selection
            exclude_candidates = [
                col for col in parcel_gdf.columns
                if col not in ['parcel_name', 'owner', 'total_acres', 'buildable_area', 'buildable_pct', 'state', 'geometry']
                and parcel_gdf[col].sum() > 0
            ]

            st.sidebar.subheader("🚫 Active Constraints")
            exclude_constraints = st.sidebar.multiselect(
                "Select constraints to apply",
                exclude_candidates,
                default=exclude_candidates[:3] if exclude_candidates else [],
                help="Constraints will use specific setback rules defined above"
            )

            # Display constraint info
            if exclude_constraints:
                st.sidebar.info(f"Applying {len(exclude_constraints)} constraint types with specific setback rules")

            # Base Map
            st.subheader("🗺️ Site Map & Constraints")
            overlay_gdf = st.session_state.get("overlay_metrics")
            base_map = render_parcel_map(parcel_gdf, overlay_gdf=overlay_gdf)
            folium_static(base_map, width=1200, height=500)

            # Layout Optimization
            st.header("⚡ Layout Optimization")
            
            if ai_enabled:
                st.subheader("🤖 AI-Powered Scenarios")
                if st.button("🚀 Generate AI-Optimized Layouts", type="primary"):
                    with st.spinner("🧠 Running AI optimization with your specifications..."):
                        
                        # Base parameters
                        base_params = {
                            'setback': setback,
                            'tracker_length': tracker_length,
                            'tracker_width': tracker_width,
                            'orientation': orientation
                        }
                        
                        # Generate AI scenarios using YOUR formulas
                        layout_scenarios = generate_ai_scenarios(
                            parcel_gdf, 
                            base_params, 
                            exclude_constraints
                        )
                        
                        st.success("✨ AI optimization complete using your capacity formulas!")
                
                # Display scenario comparison
                if layout_scenarios:
                    st.subheader("📊 Scenario Comparison")
                    
                    # Create comparison table
                    comparison_data = []
                    for scenario_name, data in layout_scenarios.items():
                        metrics = data['metrics']
                        comparison_data.append({
                            'Scenario': scenario_name,
                            'AC Capacity (MW)': metrics.get('estimated_ac_capacity_mw', 0),
                            'DC Capacity (MW)': metrics.get('estimated_dc_capacity_mw', 0),
                            'Total Tables': metrics.get('total_tables', 0),
                            'Total Modules': metrics.get('total_modules', 0),
                            'Tracker Rows': metrics.get('total_tracker_rows', 0),
                            'DC/AC Ratio': metrics.get('dcac_ratio', 0),
                            'GCR': metrics.get('gcr', 0),
                            'Nominal Check': metrics.get('nominal_capacity_check', 0),
                            'Land Use (%)': metrics.get('land_utilization_pct', 0)
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display side by side
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        # Capacity comparison chart
                        fig = px.bar(
                            comparison_df, 
                            x='Scenario', 
                            y='AC Capacity (MW)',
                            title="AC Capacity by Scenario",
                            color='AC Capacity (MW)',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Scenario selection for detailed view
                    selected_scenario = st.selectbox(
                        "Select scenario for detailed analysis:",
                        list(layout_scenarios.keys()),
                        index=0
                    )
                    
                    if selected_scenario:
                        scenario_data = layout_scenarios[selected_scenario]
                        layout_gdf = scenario_data['layout_gdf']
                        metrics = scenario_data['metrics']
                        
                        # Detailed metrics
                        st.subheader(f"📈 {selected_scenario} - Analysis")
                        
                        # Key metrics cards
                        metric_cols = st.columns(5)
                        
                        with metric_cols[0]:
                            st.metric(
                                "💡 AC Capacity", 
                                f"{metrics.get('estimated_ac_capacity_mw', 0)} MW",
                                delta=f"DC: {metrics.get('estimated_dc_capacity_mw', 0)} MW"
                            )
                        
                        with metric_cols[1]:
                            st.metric(
                                "🔧 Equipment", 
                                f"{metrics.get('total_tables', 0)} Tables",
                                delta=f"{metrics.get('total_tracker_rows', 0)} Rows"
                            )
                        
                        with metric_cols[2]:
                            st.metric(
                                "📱 Modules", 
                                f"{metrics.get('total_modules', 0):,}",
                                delta="Illuminate 650W"
                            )
                        
                        with metric_cols[3]:
                            st.metric(
                                "⚖️ DC/AC Ratio", 
                                f"{metrics.get('dcac_ratio', 0):.2f}",
                                delta=f"GCR: {metrics.get('gcr', 0):.3f}"
                            )
                        
                        with metric_cols[4]:
                            st.metric(
                                "✅ Nominal Check", 
                                f"{metrics.get('nominal_capacity_check', 0)} MW",
                                delta=f"Land Use: {metrics.get('land_utilization_pct', 0)}%"
                            )
                        
                        # Layout visualization with modules
                        st.subheader("🗺️ Optimized Layout with Modules")
                        if not layout_gdf.empty:
                            layout_map = render_parcel_map(
                                parcel_gdf, 
                                overlay_gdf=overlay_gdf, 
                                layout_gdf=layout_gdf
                            )
                            folium_static(layout_map, width=1200, height=600)
                            
                            # Show module information
                            st.info("🔍 Blue lines show tracker rows. Red dots show individual 650W modules.")
                        else:
                            st.warning("No layout generated for this scenario")
                        
                        # Technical summary using YOUR formulas
                        st.subheader("📊 Technical Summary")
                        
                        tech_data = {
                            'Parameter': [
                                'DC Capacity (MW)', 
                                'AC Capacity (MW)',
                                'DC/AC Ratio',
                                'Total Tracker Tables',
                                'Total Modules', 
                                'Module Type',
                                'Strings per Table',
                                'GCR',
                                'Nominal Capacity Check (MW)',
                                'Land Utilization (%)',
                                'Constraint Rules Applied'
                            ],
                            'Value': [
                                f"{metrics.get('estimated_dc_capacity_mw', 0)}",
                                f"{metrics.get('estimated_ac_capacity_mw', 0)}",
                                f"{metrics.get('dcac_ratio', 0):.2f}",
                                f"{metrics.get('total_tables', 0):,}",
                                f"{metrics.get('total_modules', 0):,}",
                                f"{metrics.get('module_specs', 'Illuminate 650W')}",
                                "4 strings × 27 modules",
                                f"{metrics.get('gcr', 0):.3f}",
                                f"{metrics.get('nominal_capacity_check', 0)} MW",
                                f"{metrics.get('land_utilization_pct', 0)}%",
                                "✅ Applied" if metrics.get('constraint_rules_applied') else "❌ Not Applied"
                            ]
                        }
                        
                        tech_df = pd.DataFrame(tech_data)
                        st.dataframe(tech_df, use_container_width=True, hide_index=True)
                        
                        # Export options
                        st.subheader("📥 Export Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if not layout_gdf.empty:
                                geojson = layout_gdf.to_json()
                                st.download_button(
                                    "📄 Download Layout GeoJSON",
                                    geojson,
                                    file_name=f"{selected_scenario.lower().replace(' ', '_')}_layout.geojson",
                                    mime="application/geo+json"
                                )
                        
                        with col2:
                            metrics_df = pd.DataFrame([metrics])
                            csv = metrics_df.to_csv(index=False)
                            st.download_button(
                                "📊 Download Metrics CSV",
                                csv,
                                file_name=f"{selected_scenario.lower().replace(' ', '_')}_metrics.csv",
                                mime="text/csv"
                            )
                        
                        with col3:
                            report = f"""# Solar Layout Report - {selected_scenario}

## Project Summary
- **Total Area**: {total_acres:.1f} acres
- **Buildable Area**: {buildable_acres:.1f} acres
- **Nominal Capacity Check**: {metrics.get('nominal_capacity_check', 0)} MW

## Equipment Configuration (Illuminate 650W)
- **Total Tracker Tables**: {metrics.get('total_tables', 0):,}
- **Total Modules**: {metrics.get('total_modules', 0):,}
- **Strings per Table**: 4 strings × 27 modules = 108 modules per tracker

## Capacity & Performance
- **DC Capacity**: {metrics.get('estimated_dc_capacity_mw', 0)} MW
- **AC Capacity**: {metrics.get('estimated_ac_capacity_mw', 0)} MW  
- **DC/AC Ratio**: {metrics.get('dcac_ratio', 0):.2f}

## Layout Details
- **GCR**: {metrics.get('gcr', 0):.3f}
- **Land Utilization**: {metrics.get('land_utilization_pct', 0)}%

## Constraint Rules Applied
- Parcel Boundary Setback: 50 ft
- Riverine Setback: 50 ft  
- Road Gaps: Primary 30ft, Secondary 8ft
- Wetlands Buffer: 100 ft
- Floodplain Buffer: 25 ft

## AI Optimization
Generated using your exact capacity calculation formulas and constraint setback rules.
"""
                            
                            st.download_button(
                                "📋 Download Report",
                                report,
                                file_name=f"{selected_scenario.lower().replace(' ', '_')}_report.md",
                                mime="text/markdown"
                            )
            
            else:
                # Standard optimization
                st.subheader("📐 Standard Layout")
                if st.button("Generate Standard Layout"):
                    with st.spinner("Generating layout..."):
                        layout_gdf = optimize_layout(
                            parcel_gdf,
                            setback=setback,
                            tracker_length=tracker_length,
                            tracker_width=tracker_width,
                            gcr=0.4,  # Default GCR
                            orientation=orientation,
                            exclude_columns=exclude_constraints
                        )
                        
                        if layout_gdf.empty:
                            st.error("❌ No valid tracker rows generated.")
                        else:
                            st.success(f"✅ Generated {len(layout_gdf)} tracker rows.")
                            
                            layout_map = render_parcel_map(parcel_gdf, overlay_gdf=overlay_gdf, layout_gdf=layout_gdf)
                            folium_static(layout_map, width=1200, height=600)
                            
                            geojson = layout_gdf.to_json()
                            st.download_button(
                                "📥 Download Layout",
                                geojson,
                                file_name="standard_layout.geojson",
                                mime="application/geo+json"
                            )

        except Exception as e:
            st.error(f"❌ Error processing KML: {e}")
            st.exception(e)

else:
    # Welcome page
    st.markdown("""
    ## 🚀 Welcome to AI Solar Layout Optimizer
    
    Optimized for utility-scale solar projects using **your exact specifications**:
    
    ### 🔧 Module Configuration:
    - **Illuminate 650W** modules (7.8 ft × 6.9 ft)
    - **Table Length**: 153 ft (46.6 m)
    - **4 strings × 27 modules** = 108 modules per tracker
    - **0.070244 MW DC per tracker** ✅ (CORRECTED)
    - **Expected Capacity**: 0.5 - 1.0 MW/acre (realistic range)
    
    ### 🚫 Enhanced Constraint Processing:
    - **Automatic Overlay Extraction**: Finds all constraint overlays in KML
    - **Smart Parcel Matching**: Calculates intersection areas automatically  
    - **Specific Setback Rules**: Applied based on constraint type
    - **Parcel Boundary**: 50 ft setback
    - **Riverine**: 50 ft setback
    - **Primary Roads**: 30 ft gap
    - **Secondary Roads**: 8 ft gap  
    - **Wetlands**: 100 ft buffer
    - **Floodplain**: 25 ft buffer
    
    ### 🤖 AI Scenarios:
    - **Max Capacity**: Uses secondary road gaps (8 ft) and DC/AC = 1.1
    - **Max Efficiency**: Uses primary road gaps (30 ft) and DC/AC = 1.3
    - **Balanced**: Uses average gaps (19 ft) and DC/AC = 1.2
    - **Constraint Optimized**: Adaptive spacing and DC/AC = 1.25
    
    ### 📐 Capacity Calculations:
    Uses **your exact nominal capacity formulas**:
    - GCR = DC/AC ÷ 4
    - Pitch = Module Width ÷ GCR  
    - Area per Tracker = Table Length × Pitch
    - Adjusted Area = (Buildable - 20 acres) × 95%
    - **Expected Result**: ~1,800-2,300 MW for 3,127 acres (not 11,000+ MW)
    
    Upload your Anderson KML file to start optimizing!
    """)

st.markdown("---")
st.markdown("🔆 **AI Solar Layout Optimizer** | Fixed Capacity Calculations & Enhanced Constraint Processing")