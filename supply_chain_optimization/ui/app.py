"""
Supply Chain Optimization Web UI

Run with: streamlit run supply_chain_optimization/ui/app.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from supply_chain_optimization.data.openrouteservice_client import (
    Location, SAMPLE_INDIAN_CITIES, OpenRouteServiceClient, compute_ors_distance_matrices,
    geocode_address
)
from supply_chain_optimization.data.models import (
    Supplier, Warehouse, Customer, SupplyChainNetwork
)
from supply_chain_optimization.config import create_default_config, create_small_config
from supply_chain_optimization.hybrid.coordinator import HybridSolver
from supply_chain_optimization.data.scenarios import generate_demand_scenarios

# Page config
st.set_page_config(
    page_title="Supply Chain Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .route-card {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'suppliers' not in st.session_state:
    st.session_state.suppliers = [
        {"name": loc.name, "lat": loc.lat, "lng": loc.lng, "capacity": 1000}
        for loc in SAMPLE_INDIAN_CITIES["suppliers"]
    ]

if 'warehouses' not in st.session_state:
    st.session_state.warehouses = [
        {"name": loc.name, "lat": loc.lat, "lng": loc.lng, "capacity": 5000, "holding_cost": 1.0}
        for loc in SAMPLE_INDIAN_CITIES["warehouses"]
    ]

if 'customers' not in st.session_state:
    st.session_state.customers = [
        {"name": loc.name, "lat": loc.lat, "lng": loc.lng, "demand": 100 if i == 0 else 50, "is_factory": i == 0}
        for i, loc in enumerate(SAMPLE_INDIAN_CITIES["customers"])
    ]

if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None

if 'network' not in st.session_state:
    st.session_state.network = None

# Cost settings defaults
if 'cost_settings' not in st.session_state:
    st.session_state.cost_settings = {
        "transport_cost_per_km": 2.5,
        "truck_capacity": 100,
        "shortage_penalty": 50.0,
        "holding_cost": 1.0,
        "num_scenarios": 2,
        "horizon_days": 7
    }


def create_map(suppliers, warehouses, customers, routes=None):
    """Create a Folium map with all locations and routes."""
    # Center map on India
    center_lat = np.mean([s["lat"] for s in suppliers + warehouses + customers])
    center_lng = np.mean([s["lng"] for s in suppliers + warehouses + customers])
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=5, tiles="cartodbpositron")
    
    # Add supplier markers (blue)
    for s in suppliers:
        folium.Marker(
            location=[s["lat"], s["lng"]],
            popup=f"<b>SUPPLIER</b><br>{s['name']}<br>Capacity: {s['capacity']} units",
            icon=folium.Icon(color="blue", icon="industry", prefix="fa"),
            tooltip=s["name"]
        ).add_to(m)
    
    # Add warehouse markers (green)
    for w in warehouses:
        folium.Marker(
            location=[w["lat"], w["lng"]],
            popup=f"<b>WAREHOUSE</b><br>{w['name']}<br>Capacity: {w['capacity']} units",
            icon=folium.Icon(color="green", icon="warehouse", prefix="fa"),
            tooltip=w["name"]
        ).add_to(m)
    
    # Add customer markers (red/orange)
    for c in customers:
        color = "red" if c.get("is_factory", False) else "orange"
        icon = "building" if c.get("is_factory", False) else "user"
        label = "FACTORY" if c.get("is_factory", False) else "CUSTOMER"
        folium.Marker(
            location=[c["lat"], c["lng"]],
            popup=f"<b>{label}</b><br>{c['name']}<br>Demand: {c['demand']} units",
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
            tooltip=c["name"]
        ).add_to(m)
    
    # Add route lines if provided
    if routes:
        # Supplier -> Warehouse routes (blue lines)
        for route in routes.get("sw", []):
            folium.PolyLine(
                locations=[[route["from_lat"], route["from_lng"]], [route["to_lat"], route["to_lng"]]],
                color="blue",
                weight=3,
                opacity=0.7,
                popup=f"{route['from_name']} → {route['to_name']}<br>{route['distance']:.0f} km"
            ).add_to(m)
        
        # Warehouse -> Customer routes (green lines)
        for route in routes.get("wc", []):
            folium.PolyLine(
                locations=[[route["from_lat"], route["from_lng"]], [route["to_lat"], route["to_lng"]]],
                color="green",
                weight=3,
                opacity=0.7,
                popup=f"{route['from_name']} → {route['to_name']}<br>{route['distance']:.0f} km"
            ).add_to(m)
    
    # Add legend with dark theme support
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background: rgba(30, 30, 30, 0.9); color: white;
                padding: 12px 15px; border-radius: 8px; font-size: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.4); border: 1px solid #444;">
        <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px;">Legend</div>
        <div style="margin: 4px 0;"><span style="color: #3498db; font-size: 14px;">●</span> Supplier</div>
        <div style="margin: 4px 0;"><span style="color: #2ecc71; font-size: 14px;">●</span> Warehouse</div>
        <div style="margin: 4px 0;"><span style="color: #e74c3c; font-size: 14px;">●</span> Factory</div>
        <div style="margin: 4px 0;"><span style="color: #f39c12; font-size: 14px;">●</span> Customer</div>
        <div style="margin: 4px 0; border-top: 1px solid #555; padding-top: 4px;"><span style="color: #3498db;">━━</span> Supplier→Warehouse</div>
        <div style="margin: 4px 0;"><span style="color: #2ecc71;">━━</span> Warehouse→Customer</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def run_optimization():
    """Run the supply chain optimization."""
    with st.spinner("Running optimization..."):
        # Create network from session state
        config = create_small_config()
        config.network.num_suppliers = len(st.session_state.suppliers)
        config.network.num_warehouses = len(st.session_state.warehouses)
        config.network.num_customers = len(st.session_state.customers)
        
        # Apply user cost settings
        cost_settings = st.session_state.cost_settings
        config.costs.variable_cost_per_km = cost_settings["transport_cost_per_km"]
        config.capacity.truck_capacity = cost_settings["truck_capacity"]
        config.costs.shortage_penalty_default = cost_settings["shortage_penalty"]
        config.costs.holding_cost_default = cost_settings["holding_cost"]
        config.scenarios.num_scenarios = int(cost_settings["num_scenarios"])
        config.scenarios.horizon_days = int(cost_settings["horizon_days"])
        
        # Build entities
        suppliers = [
            Supplier(
                id=i,
                name=s["name"],
                location=(s["lat"], s["lng"]),
                capacity={t: s["capacity"] for t in range(config.scenarios.horizon_days)}
            )
            for i, s in enumerate(st.session_state.suppliers)
        ]
        
        warehouses = [
            Warehouse(
                id=j,
                name=w["name"],
                location=(w["lat"], w["lng"]),
                capacity=w["capacity"],
                holding_cost=w.get("holding_cost", 1.0),
                initial_inventory=w["capacity"] * 0.2
            )
            for j, w in enumerate(st.session_state.warehouses)
        ]
        
        customers = []
        for k, c in enumerate(st.session_state.customers):
            is_factory = c.get("is_factory", k == 0)
            tier = 0 if is_factory else (1 if k <= len(st.session_state.customers)//4 else 2)
            customers.append(Customer(
                id=k,
                name=c["name"],
                location=(c["lat"], c["lng"]),
                is_factory=is_factory,
                shortage_penalty=10000.0 if is_factory else 50.0,
                service_level_z=3.0 if is_factory else 1.96,
                tier=tier,
                average_demand=c["demand"]
            ))
        
        network = SupplyChainNetwork(
            suppliers=suppliers,
            warehouses=warehouses,
            customers=customers
        )
        
        # Compute distances using ORS
        try:
            supplier_coords = [(s["lat"], s["lng"]) for s in st.session_state.suppliers]
            warehouse_coords = [(w["lat"], w["lng"]) for w in st.session_state.warehouses]
            customer_coords = [(c["lat"], c["lng"]) for c in st.session_state.customers]
            
            compute_ors_distance_matrices(network, supplier_coords, warehouse_coords, customer_coords)
        except Exception as e:
            st.warning(f"Using Euclidean distances (ORS error: {e})")
            from supply_chain_optimization.data.distances import compute_distance_matrices, compute_lead_time_matrices
            compute_distance_matrices(network)
            compute_lead_time_matrices(network)
        
        # Generate scenarios
        scenarios = generate_demand_scenarios(
            network,
            config.scenarios,
            random_seed=config.random_seed
        )
        
        # Run hybrid solver
        solver = HybridSolver(network, config, scenarios)
        result = solver.solve(use_classical_stage1=True)
        
        st.session_state.network = network
        st.session_state.optimization_result = result
        st.session_state.solver = solver
        
        return result


def get_routes_from_result(network, result):
    """Extract routes from optimization result."""
    routes = {"sw": [], "wc": []}
    
    # Supplier -> Warehouse routes
    for (i, j), val in result.x_assign.items():
        if val == 1:
            supplier = network.suppliers[i]
            warehouse = network.warehouses[j]
            distance = network.distance_sw[i, j] if network.distance_sw is not None else 0
            routes["sw"].append({
                "from_name": supplier.name,
                "from_lat": supplier.location[0],
                "from_lng": supplier.location[1],
                "to_name": warehouse.name,
                "to_lat": warehouse.location[0],
                "to_lng": warehouse.location[1],
                "distance": distance
            })
    
    # Warehouse -> Customer routes
    for (j, k), val in result.y_assign.items():
        if val == 1:
            warehouse = network.warehouses[j]
            customer = network.customers[k]
            distance = network.distance_wc[j, k] if network.distance_wc is not None else 0
            routes["wc"].append({
                "from_name": warehouse.name,
                "from_lat": warehouse.location[0],
                "from_lng": warehouse.location[1],
                "to_name": customer.name,
                "to_lat": customer.location[0],
                "to_lng": customer.location[1],
                "distance": distance
            })
    
    return routes


# === MAIN UI ===

st.markdown('<h1 class="main-header">🚚 Supply Chain Optimizer</h1>', unsafe_allow_html=True)
st.markdown("*Quantum-enabled two-stage stochastic optimization with real road distances*")

# Sidebar for inputs
with st.sidebar:
    st.header("📍 Network Configuration")
    
    # Suppliers
    with st.expander("🔵 Suppliers", expanded=False):
        st.caption("Add/edit suppliers")
        
        # Display current suppliers
        for i, s in enumerate(st.session_state.suppliers):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{s['name']}")
            with col2:
                if st.button("❌", key=f"del_s_{i}"):
                    st.session_state.suppliers.pop(i)
                    st.rerun()
        
        # Add new supplier
        st.markdown("---")
        new_s_name = st.text_input("New supplier name", key="new_supplier_name")
        new_s_address = st.text_input("Address (e.g. 'Mumbai, India')", key="new_s_address", 
                                       placeholder="Type address and click Add...")
        new_s_cap = st.number_input("Capacity", value=1000, key="new_s_cap")
        
        if st.button("+ Add Supplier"):
            if new_s_name and new_s_address:
                with st.spinner("Finding location..."):
                    coords = geocode_address(new_s_address)
                    if coords:
                        st.session_state.suppliers.append({
                            "name": new_s_name, "lat": coords[0], "lng": coords[1], 
                            "capacity": new_s_cap, "address": new_s_address
                        })
                        st.rerun()
                    else:
                        st.error("Could not find address. Try being more specific.")
            else:
                st.warning("Please enter name and address")
    
    # Warehouses
    with st.expander("🟢 Warehouses", expanded=False):
        st.caption("Add/edit warehouses")
        
        for i, w in enumerate(st.session_state.warehouses):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{w['name']}")
            with col2:
                if st.button("❌", key=f"del_w_{i}"):
                    st.session_state.warehouses.pop(i)
                    st.rerun()
        
        st.markdown("---")
        new_w_name = st.text_input("New warehouse name", key="new_warehouse_name")
        new_w_address = st.text_input("Address (e.g. 'Pune, India')", key="new_w_address",
                                       placeholder="Type address and click Add...")
        new_w_cap = st.number_input("Capacity", value=5000, key="new_w_cap")
        
        if st.button("+ Add Warehouse"):
            if new_w_name and new_w_address:
                with st.spinner("Finding location..."):
                    coords = geocode_address(new_w_address)
                    if coords:
                        st.session_state.warehouses.append({
                            "name": new_w_name, "lat": coords[0], "lng": coords[1], 
                            "capacity": new_w_cap, "holding_cost": 1.0, "address": new_w_address
                        })
                        st.rerun()
                    else:
                        st.error("Could not find address. Try being more specific.")
            else:
                st.warning("Please enter name and address")
    
    # Customers
    with st.expander("🟠 Customers", expanded=False):
        st.caption("Add/edit customers (first one is factory)")
        
        for i, c in enumerate(st.session_state.customers):
            col1, col2 = st.columns([3, 1])
            with col1:
                label = "🏭" if c.get("is_factory", i == 0) else ""
                st.text(f"{label} {c['name']}")
            with col2:
                if st.button("❌", key=f"del_c_{i}"):
                    st.session_state.customers.pop(i)
                    st.rerun()
        
        st.markdown("---")
        new_c_name = st.text_input("New customer name", key="new_customer_name")
        new_c_address = st.text_input("Address (e.g. 'Jaipur, India')", key="new_c_address",
                                       placeholder="Type address and click Add...")
        new_c_demand = st.number_input("Demand", value=50, key="new_c_demand")
        new_c_factory = st.checkbox("Is Factory?", key="new_c_factory")
        
        if st.button("+ Add Customer"):
            if new_c_name and new_c_address:
                with st.spinner("Finding location..."):
                    coords = geocode_address(new_c_address)
                    if coords:
                        st.session_state.customers.append({
                            "name": new_c_name, "lat": coords[0], "lng": coords[1],
                            "demand": new_c_demand, "is_factory": new_c_factory, "address": new_c_address
                        })
                        st.rerun()
                    else:
                        st.error("Could not find address. Try being more specific.")
            else:
                st.warning("Please enter name and address")
    
    st.markdown("---")
    
    # Cost Settings
    with st.expander("💰 Cost Settings", expanded=False):
        st.caption("Configure optimization parameters")
        
        st.session_state.cost_settings["transport_cost_per_km"] = st.number_input(
            "Transport Cost (₹/km)", 
            value=st.session_state.cost_settings["transport_cost_per_km"],
            min_value=0.1, key="cost_transport"
        )
        st.session_state.cost_settings["truck_capacity"] = st.number_input(
            "Truck Capacity (units)", 
            value=st.session_state.cost_settings["truck_capacity"],
            min_value=1, key="cost_truck_cap"
        )
        st.session_state.cost_settings["shortage_penalty"] = st.number_input(
            "Shortage Penalty (₹/unit)", 
            value=st.session_state.cost_settings["shortage_penalty"],
            min_value=1.0, key="cost_shortage"
        )
        st.session_state.cost_settings["holding_cost"] = st.number_input(
            "Holding Cost (₹/unit/day)", 
            value=st.session_state.cost_settings["holding_cost"],
            min_value=0.1, key="cost_holding"
        )
        st.session_state.cost_settings["num_scenarios"] = st.number_input(
            "Demand Scenarios", 
            value=st.session_state.cost_settings["num_scenarios"],
            min_value=1, max_value=5, key="cost_scenarios"
        )
        st.session_state.cost_settings["horizon_days"] = st.number_input(
            "Planning Horizon (days)", 
            value=st.session_state.cost_settings["horizon_days"],
            min_value=1, max_value=30, key="cost_horizon"
        )
    
    st.markdown("---")
    
    # Run optimization button
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        run_optimization()
        st.rerun()
    
    # Reset button
    if st.button("🔄 Reset to Default", use_container_width=True):
        for key in ['suppliers', 'warehouses', 'customers', 'optimization_result', 'network', 'cost_settings']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# Main content area
col_map, col_results = st.columns([3, 2])

with col_map:
    st.subheader("📍 Network Map")
    
    # Get routes if optimization has run
    routes = None
    if st.session_state.optimization_result and st.session_state.network:
        routes = get_routes_from_result(st.session_state.network, st.session_state.optimization_result)
    
    # Create and display map
    m = create_map(
        st.session_state.suppliers,
        st.session_state.warehouses,
        st.session_state.customers,
        routes
    )
    st_folium(m, width=700, height=500)

with col_results:
    st.subheader("📊 Results")
    
    if st.session_state.optimization_result:
        result = st.session_state.optimization_result
        network = st.session_state.network
        
        # Summary metrics - handle infinity/large values
        col1, col2, col3 = st.columns(3)
        
        # Calculate actual transport cost for display
        total_km = 0
        for (i, j), val in result.x_assign.items():
            if val == 1 and network.distance_sw is not None:
                total_km += network.distance_sw[i, j]
        for (j, k), val in result.y_assign.items():
            if val == 1 and network.distance_wc is not None:
                total_km += network.distance_wc[j, k]
        
        transport_cost = total_km * st.session_state.cost_settings["transport_cost_per_km"]
        
        # Check for infinity
        if np.isinf(result.total_objective) or result.total_objective > 1e12:
            col1.metric("Transport Cost", f"₹{transport_cost/1000:.1f}K")
        else:
            col1.metric("Total Cost", f"₹{result.total_objective/1e6:.1f}M")
        
        col2.metric("Supplier Routes", sum(1 for v in result.x_assign.values() if v == 1))
        col3.metric("Customer Routes", sum(1 for v in result.y_assign.values() if v == 1))
        
        # Show total distance
        st.info(f"📏 Total Route Distance: **{total_km:.0f} km** | Transport Cost: **₹{transport_cost:,.0f}**")
        
        # Routes table
        st.markdown("#### Supplier → Warehouse Routes")
        sw_data = []
        for (i, j), val in result.x_assign.items():
            if val == 1:
                sw_data.append({
                    "Supplier": network.suppliers[i].name,
                    "Warehouse": network.warehouses[j].name,
                    "Distance (km)": f"{network.distance_sw[i, j]:.0f}",
                    "Lead Time": f"{network.lead_time_sw[i, j]:.0f} days"
                })
        if sw_data:
            st.dataframe(pd.DataFrame(sw_data), use_container_width=True, hide_index=True)
        
        st.markdown("#### Warehouse → Customer Routes")
        wc_data = []
        for (j, k), val in result.y_assign.items():
            if val == 1:
                customer = network.customers[k]
                label = "🏭" if customer.is_factory else ""
                wc_data.append({
                    "Warehouse": network.warehouses[j].name,
                    "Customer": f"{label} {customer.name}",
                    "Distance (km)": f"{network.distance_wc[j, k]:.0f}",
                    "Lead Time": f"{network.lead_time_wc[j, k]:.0f} days"
                })
        if wc_data:
            st.dataframe(pd.DataFrame(wc_data), use_container_width=True, hide_index=True)
        
        # Cost breakdown chart - show transport cost by route type
        st.markdown("#### Cost Breakdown by Route Type")
        
        # Calculate costs by route type
        sw_km = sum(network.distance_sw[i, j] for (i, j), v in result.x_assign.items() 
                    if v == 1 and network.distance_sw is not None)
        wc_km = sum(network.distance_wc[j, k] for (j, k), v in result.y_assign.items() 
                    if v == 1 and network.distance_wc is not None)
        
        cost_per_km = st.session_state.cost_settings["transport_cost_per_km"]
        sw_cost = sw_km * cost_per_km
        wc_cost = wc_km * cost_per_km
        
        cost_breakdown = {
            "Supplier→Warehouse": sw_cost,
            "Warehouse→Customer": wc_cost
        }
        
        if sw_cost > 0 or wc_cost > 0:
            fig = px.pie(
                values=list(cost_breakdown.values()),
                names=list(cost_breakdown.keys()),
                title=f"Transport Cost Distribution (₹{sw_cost + wc_cost:,.0f} total)",
                color_discrete_sequence=["#3498db", "#2ecc71"]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        with st.expander("ℹ️ What do these costs mean?"):
            st.markdown("""
            - **Supplier→Warehouse**: Cost to transport goods from suppliers to warehouses
            - **Warehouse→Customer**: Cost to deliver goods from warehouses to customers
            
            The optimization finds the best assignment of routes to minimize total cost while ensuring:
            1. Each customer is served by a warehouse
            2. Each warehouse is supplied by suppliers
            3. Factory has backup warehouses (redundancy)
            """)
        
    else:
        st.info("👆 Click 'Run Optimization' to see results")
        
        # Show current network summary
        st.markdown("#### Current Network")
        st.write(f"- **Suppliers:** {len(st.session_state.suppliers)}")
        st.write(f"- **Warehouses:** {len(st.session_state.warehouses)}")
        st.write(f"- **Customers:** {len(st.session_state.customers)}")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Supply Chain Optimization with Quantum-Classical Hybrid Algorithm | "
    "Distances powered by OpenRouteService"
    "</div>",
    unsafe_allow_html=True
)
