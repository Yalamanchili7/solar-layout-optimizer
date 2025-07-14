import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import joblib
from math import sqrt, log

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

MODULE_FILE = "data/Module_and_CapEx_250409.xlsx"
MODEL_PATH = os.path.join("models", "sy_gbdt_model.pkl")
SCALER_PATH = os.path.join("models", "sy_scaler.pkl")

def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def transform_features(ghi, cf, dcac):
    ghi_cf = ghi * cf
    ghi_dcac = ghi * dcac
    cf_dcac = cf * dcac
    log_dcac = log(dcac + 2)
    sqrt_ghi = sqrt(ghi)
    return np.array([[ghi, cf, dcac, ghi_cf, ghi_dcac, cf_dcac, log_dcac, sqrt_ghi]])

def predict_sy_range(ghi, cf, dcac=1.3):
    model, scaler = load_model_and_scaler()
    X = transform_features(ghi, cf, dcac)
    X_scaled = scaler.transform(X)
    return round(model.predict(X_scaled)[0], 5)

def get_uploaded_buildable_acres():
    return st.session_state.get("total_buildable_area", 300.0)

def get_uploaded_tgy_ghi():
    return st.session_state.get("uploaded_tgy_ghi", 2000.0)

def capacity_estimation_ui():
    # FIXED: Choose scenario - only show "All Acres" instead of "(Default)"
    all_scenarios = {}
    
    # Add "All Acres" scenario if default_scenario exists
    if "default_scenario" in st.session_state:
        default_df = st.session_state["default_scenario"]
        if not default_df.empty:
            all_scenarios["All Acres"] = default_df
    
    # Add saved scenarios
    all_scenarios.update(st.session_state.get("saved_scenarios", {}))

    selected_scenario_name = st.selectbox("Select Scenario", list(all_scenarios.keys()))
    selected_scenario_df = all_scenarios[selected_scenario_name]
    st.caption("Please save your scenarios at the end of this page to move forward with the CapEx")

    if selected_scenario_df.empty:
     st.warning("No parcels available in this scenario.")
     return

    
    mode = st.radio("Select Mode", ["Nominal Capacity", "Capacity Validation"])

    uploaded_acres = selected_scenario_df["buildable_area"].sum()
    use_kml_toggle = st.toggle("Use Buildable Area from KML", value=True)
    base_acreage = uploaded_acres if use_kml_toggle and uploaded_acres else 300.0
    subtract_substation = st.toggle("Include 20 acres for Substation/BESS?", value=True)
    adjusted_area = (uploaded_acres - 20) * 0.95 if subtract_substation else uploaded_acres * 0.95
    uploaded_ghi = get_uploaded_tgy_ghi()

    st.markdown(f"**Buildable Area (acres):** `{base_acreage:.2f}`")
    st.markdown(f"**Adjusted Buildable Area (acres):** `{adjusted_area:.2f}`")
    st.info("Adjusted Buildable Area = [Base Acres - (BESS/Substation) * 95% Availablity]")
    st.markdown(f"**GHI:** `{uploaded_ghi:.2f}`")

    df = pd.read_excel(MODULE_FILE, sheet_name="Module Database")
    cod_column = [col for col in df.columns if "cod" in col.lower()][0]
    df[cod_column] = df[cod_column].astype(int)
    cod_options = sorted(df[cod_column].dropna().unique())
    selected_cod = st.selectbox("Project COD", cod_options, index=len(cod_options) - 1)
    st.caption("Select COD for Module Specifications")
    

    row = df[df["COD:"] == selected_cod].iloc[0]
    mod_width = row["Mod Width (ft):"]
    rack_length = row["Total Rack Length (ft):"]
    mw_dc_per_tracker = row["MW DC per Tracker:"]

    dcac_range = np.round(np.arange(1.1, 1.41, 0.05), 2)

    if mode == "Nominal Capacity":
        st.subheader("📐 Nominal Capacity Mode")
        st.caption(
           "Estimate the maximum AC & DC capacity the available land can support assuming optimal design (CF ≈ 1.0) "
           "by varying DC/AC ratio and computing resulting GCR, tracker count, and capacity."
        )

        cf_fixed = 1.0
        results = []

        for dcac in dcac_range:
            gcr = (dcac *cf_fixed) / 4
            pitch = mod_width / gcr
            area_tracker_ft2 = rack_length * pitch
            area_tracker_ac = area_tracker_ft2 / 43560
            tracker_count = adjusted_area / area_tracker_ac
            dc_capacity = tracker_count * mw_dc_per_tracker
            ac_capacity = dc_capacity / dcac
            sy = predict_sy_range(uploaded_ghi, cf_fixed, dcac)

            results.append({
                "DC/AC Ratio": round(dcac, 2),
                "GCR": gcr,
                "GCR (%)": gcr * 100,
                "Pitch (ft)": pitch,
                "Area per Tracker (ft²)": area_tracker_ft2,
                "Area per Tracker (acres)": area_tracker_ac,
                "Trackers": tracker_count,
                "DC Capacity (MW)": dc_capacity,
                "AC Capacity (MW)": ac_capacity,
                "SY (kWh/kWp)": sy
            })
        
        if results:
           first_ac_capacity = results[0]["AC Capacity (MW)"]
           st.markdown(
              f"<h3 style='color:#21c55d; font-weight:800;'>AC Capacity: {first_ac_capacity:.2f} MW</h3>",
              unsafe_allow_html=True
            )

        with st.expander("📘 Engineering Assumptions & Formulas"):
            st.markdown(f"""
            - **Constraint Factor (CF):** Fixed at `1.0`
            - **GCR and DC/AC Relationship:**
              - CF = (4 * GCR) ÷ DC/AC
              - With CF fixed at 1.0, rearranged: DC/AC = 4 * GCR
            - **GCR = DCAC ÷ 4**
            - **Pitch = Module Width ÷ GCR**
            - **Area per Tracker (ft²) = Rack Length × Pitch**
            - **Area per Tracker (acres) = ft² ÷ 43,560**
            - **Trackers = Adjusted Area ÷ Area per Tracker**
            - **DC Capacity = Trackers × MW per Tracker**
            - **AC Capacity = DC Capacity ÷ DC/AC**
            """)

        nominal_df = pd.DataFrame(results)
        st.markdown("### 📋 Nominal Capacity Table")
        st.dataframe(nominal_df.style.format("{:.7f}"), use_container_width=True, hide_index=True)

        st.altair_chart(
            alt.Chart(nominal_df).mark_line(point=True).encode(
                x=alt.X("GCR", scale=alt.Scale(domain=[0.20, 0.40]), title="GCR"),
                y=alt.Y("DC/AC Ratio", scale=alt.Scale(domain=[1.0, 1.5]), title="DC/AC Ratio"),
                tooltip=["GCR", "DC/AC Ratio", "GCR (%)"]
            ).properties(
               title="DC/AC Ratio vs GCR (Nominal Mode)"
            ).interactive(), 
            use_container_width=True
        )


        st.altair_chart(
            alt.Chart(nominal_df).mark_line(point=True).encode(
                x="DC/AC Ratio", y="SY (kWh/kWp)", tooltip=["DC/AC Ratio", "SY (kWh/kWp)"]
            ).properties(title="Specific Yield vs DC/AC Ratio").interactive(),
            use_container_width=True
        )

        if st.button("💾 Save Nominal Results"):
            nominal_results_data = {
                "buildable_area": base_acreage,
                "adjusted_area": adjusted_area,
                "results_table": nominal_df.to_dict("records"),
                "module_width": mod_width,
                "rack_length": rack_length,
                "dc_per_tracker": mw_dc_per_tracker,
                "ghi": uploaded_ghi,
                "source_scenario": selected_scenario_name,
                "cod_year": selected_cod
            }
            
            # FIXED: Save with scenario-specific key using consistent naming
            if selected_scenario_name == "All Acres":
                # For "All Acres", save to both keys for backward compatibility
                st.session_state["capacity_nominal_results_(Default)"] = nominal_results_data
                st.session_state["capacity_nominal_results_All Acres"] = nominal_results_data
                st.session_state["capacity_nominal_results"] = nominal_results_data
            else:
                scenario_specific_key = f"capacity_nominal_results_{selected_scenario_name}"
                st.session_state[scenario_specific_key] = nominal_results_data
            
            st.success("✅ Nominal results saved to session.")
            st.info(f"🔑 Saved for scenario: {selected_scenario_name}")

    elif mode == "Capacity Validation":
        st.subheader("🎯 Capacity Validation Mode")
        st.caption(
            "Check if a user-specified target AC capacity can fit on the available land by calculating the constraint factor (CF) "
            "and classifying how constrained the site is at a fixed DC/AC ratio."
        )
        target_mw_ac = st.number_input("Enter Target AC Capacity (MW)", min_value=0.0, value=300.0)

        dcac_fixed = 1.2
        target_mw_dc = target_mw_ac * dcac_fixed
        tracker_qty = target_mw_dc / mw_dc_per_tracker 
        area_per_tracker_acre = adjusted_area / tracker_qty
        area_per_tracker_ft = area_per_tracker_acre * 43560
        pitch = area_per_tracker_ft / rack_length
        gcr = mod_width / pitch
        constraint_factor = (4 * gcr) / dcac_fixed

        # Classification
        if constraint_factor < 0.75:
            cf_class = "Likely Excess Land"
        elif 0.75 <= constraint_factor < 0.85:
            cf_class = "Very Underconstrained"
        elif 0.85 <= constraint_factor < 1.0:
            cf_class = "Slightly Underconstrained"
        elif 1.0 <= constraint_factor <= 1.15:
            cf_class = "Slightly Overconstrained"
        elif 1.15 < constraint_factor <= 1.25:
            cf_class = "Moderately Overconstrained"
        elif 1.25 < constraint_factor <= 1.5:
            cf_class = "Heavily Overconstrained"
        else:
            cf_class = "Impractical for Construction"

        st.markdown(f"""
        **Computed DC Capacity:** `{target_mw_dc:.7f}` MW  
        **Assumed DC/AC Ratio:** `{dcac_fixed:.2f}`  
        **Derived Constraint Factor:** `{constraint_factor:.7f}`  
        **⚖️ Constraint Factor Classification:** `{cf_class}`
        """)

        with st.expander("📘 Engineering Calculation Breakdown"):
            st.markdown(f"""
            - **Target AC Capacity:** `{target_mw_ac}` MW  
            - **Assumed DC/AC Ratio:** `{dcac_fixed}`  
            - **Target DC Capacity:** `{target_mw_dc}` MW  
            - **Trackers Needed:** `{tracker_qty}`  
            - **Area per Tracker (acres):** `{area_per_tracker_acre}`  
            - **Pitch (ft):** `{pitch}`  
            - **GCR = Module Width ÷ Pitch:** `{gcr}`  
            - **Constraint Factor (CF) = (4 × GCR) ÷ DC/AC:** `{constraint_factor}`
            """)

        results = []
        for dcac in dcac_range:
            dc_capacity = target_mw_ac * dcac
            tracker_count = dc_capacity / mw_dc_per_tracker
            area_per_tracker_ac = adjusted_area / tracker_count
            area_per_tracker_ft2 = area_per_tracker_ac * 43560
            pitch = area_per_tracker_ft2 / rack_length
            gcr = mod_width / pitch
            sy = predict_sy_range(uploaded_ghi, constraint_factor, dcac)
            ac_capacity = dc_capacity / dcac

            results.append({
                "DC/AC Ratio": round(dcac, 2),
                "GCR": gcr,
                "GCR (%)": gcr * 100,
                "Pitch (ft)": pitch,
                "Area per Tracker (ft²)": area_per_tracker_ft2,
                "Area per Tracker (acres)": area_per_tracker_ac,
                "Trackers": tracker_count,
                "DC Capacity (MW)": dc_capacity,
                "AC Capacity (MW)": ac_capacity,
                "SY (kWh/kWp)": sy
            })

        validation_df = pd.DataFrame(results)
        st.markdown("### 📋 Capacity Validation Table")
        st.dataframe(validation_df.style.format("{:.7f}"), use_container_width=True, hide_index=True)

        st.altair_chart(
            alt.Chart(validation_df).mark_line(point=True).encode(
                x=alt.X("GCR", scale=alt.Scale(domain=[0.20, 0.40]), title="GCR"),
                y=alt.Y("DC/AC Ratio", scale=alt.Scale(domain=[1.0, 1.5]), title="DC/AC Ratio"),
                tooltip=["GCR", "DC/AC Ratio", "GCR (%)"]
            ).properties(
               title="DC/AC Ratio vs GCR (Validation Mode)"
            ).interactive(),
            use_container_width=True
        )


        st.altair_chart(
            alt.Chart(validation_df).mark_line(point=True).encode(
                x="DC/AC Ratio", y="SY (kWh/kWp)", tooltip=["DC/AC Ratio", "SY (kWh/kWp)"]
            ).properties(title="Specific Yield vs DC/AC Ratio (Validation Mode)").interactive(),
            use_container_width=True
        )

        if st.button("💾 Save Validation Results"):
            validation_results_data = {
                "target_mw_ac": target_mw_ac,
                "target_mw_dc": target_mw_dc,
                "adjusted_area": adjusted_area,
                "constraint_factor": constraint_factor,
                "module_width": mod_width,
                "rack_length": rack_length,
                "dc_per_tracker": mw_dc_per_tracker,
                "pitch": pitch,
                "gcr": gcr,
                "tracker_count": tracker_qty,
                "ghi": uploaded_ghi,
                "cod_year": selected_cod,
                "source_scenario": selected_scenario_name,
                "results_table": validation_df.to_dict("records")
            }
            
            # FIXED: Save with scenario-specific key using consistent naming
            if selected_scenario_name == "All Acres":
                # For "All Acres", save to both keys for backward compatibility
                st.session_state["capacity_validation_results_(Default)"] = validation_results_data
                st.session_state["capacity_validation_results_All Acres"] = validation_results_data
                st.session_state["capacity_validation_results"] = validation_results_data
            else:
                scenario_specific_key = f"capacity_validation_results_{selected_scenario_name}"
                st.session_state[scenario_specific_key] = validation_results_data
            
            st.success(f"✅ Validation results saved for scenario: {selected_scenario_name}")
            st.info(f"🔑 Saved for scenario: {selected_scenario_name}")
            
            
def compute_nominal_capacity_ac(buildable_area, mod_width, rack_length, mw_dc_per_tracker, dcac_ratio=1.2, substation_area=20, availability=0.95):
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


def compute_nominal_capacity_by_parcel(gdf, dcac_ratio=1.2, substation_area=20, availability=0.95):
    import pandas as pd
    MODULE_FILE = "data/Module_and_CapEx_250409.xlsx"
    df = pd.read_excel(MODULE_FILE, sheet_name="Module Database")

    cod_column = [col for col in df.columns if "cod" in col.lower()][0]
    df[cod_column] = df[cod_column].astype(int)
    latest_row = df[df[cod_column] == df[cod_column].max()].iloc[0]

    mod_width = latest_row["Mod Width (ft):"]
    rack_length = latest_row["Total Rack Length (ft):"]
    mw_dc_per_tracker = latest_row["MW DC per Tracker:"]

    gdf = gdf.copy()
    gdf["nominal_mwac"] = gdf["buildable_area"].apply(
        lambda area: compute_nominal_capacity_ac(area, mod_width, rack_length, mw_dc_per_tracker, dcac_ratio, substation_area, availability)
    )
    return gdf