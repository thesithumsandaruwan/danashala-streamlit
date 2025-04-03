import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mesa
import numpy as np
import random
from PIL import Image # <-- Import Pillow Image

# --- Assume model.py is in the same directory ---
try:
    from model import DiningHallModel, MonkAgent
except ImportError:
    st.error("Error: Could not import 'model.py'. Make sure it's in the same directory.")
    # Provide dummy classes/functions if needed for Streamlit to load initially
    class DiningHallModel: pass
    class MonkAgent: pass
    # You might need to define dummy versions of functions used before model init if import fails badly


# --- Page Config ---
st.set_page_config(layout="wide", page_title="Monk Crowd Simulation")

st.title("Monastery Dining Hall Simulation (Agent-Based Model - Mesa 3.x)")
st.write("Simulating monks moving along a path, stopping at dining spots, and exiting.")

# --- Simulation Parameters (Sidebar) ---
st.sidebar.header("Simulation Parameters")
N_agents = st.sidebar.slider("Number of Monks (Target)", 5, 500, 100, key="n_agents") # Added key for potential state issues
agent_speed = st.sidebar.slider("Average Agent Speed", 1.0, 5.0, 2.0, 0.1, key="agent_speed")
dining_wait_time = st.sidebar.slider("Dining Wait Time (steps)", 10, 200, 50, key="wait_time")
num_dining_spots = st.sidebar.slider("Number of Dining Spots", 1, 20, 10, key="n_spots") # Increased max a bit
max_steps = st.sidebar.number_input("Maximum Simulation Steps", 100, 5000, 1000, key="max_steps")

# --- Visualization Settings ---
st.sidebar.header("Visualization")
show_path = st.sidebar.checkbox("Show Path", True, key="show_path")
show_dining_spots = st.sidebar.checkbox("Show Dining Spots", True, key="show_spots")
agent_radius_vis = 3 # For visualization scale

# --- Background Map Settings ---
st.sidebar.header("Background Map")
uploaded_map = st.sidebar.file_uploader("Upload Map Image", type=["png", "jpg", "jpeg"], key="map_upload")

# Add default extent values (assuming grid size is known or set later)
# We'll use placeholder defaults first, then update if model exists
default_width = 600
default_height = 600
if 'model' in st.session_state and st.session_state.model:
    default_width = st.session_state.model.grid_width
    default_height = st.session_state.model.grid_height

map_left = st.sidebar.number_input("Map Left Coordinate", value=0.0, key="map_left", step=10.0)
map_right = st.sidebar.number_input("Map Right Coordinate", value=float(default_width), key="map_right", step=10.0)
map_bottom = st.sidebar.number_input("Map Bottom Coordinate", value=0.0, key="map_bottom", step=10.0)
map_top = st.sidebar.number_input("Map Top Coordinate", value=float(default_height), key="map_top", step=10.0)
map_alpha = st.sidebar.slider("Map Transparency", 0.0, 1.0, 0.5, key="map_alpha") # Default 50% transparent


# --- Simulation Execution ---
# Use session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'simulation_complete' not in st.session_state:
    st.session_state.simulation_complete = False
if 'uploaded_map_data' not in st.session_state: # Store processed map data
    st.session_state.uploaded_map_data = None

# Store uploaded file object to persist across reruns if controls are changed
if uploaded_map is not None:
    # Check if it's a new file upload
    if 'current_map_name' not in st.session_state or st.session_state.current_map_name != uploaded_map.name:
         try:
             img = Image.open(uploaded_map)
             st.session_state.uploaded_map_data = np.asarray(img)
             st.session_state.current_map_name = uploaded_map.name # Store name to detect changes
             st.sidebar.success(f"Map '{uploaded_map.name}' loaded.")
         except Exception as e:
             st.sidebar.error(f"Error loading image: {e}")
             st.session_state.uploaded_map_data = None
             st.session_state.current_map_name = None
elif 'current_map_name' in st.session_state: # Handle clearing the map
     st.session_state.uploaded_map_data = None
     st.session_state.current_map_name = None


# Button to start/restart simulation
if st.sidebar.button("Run Simulation", key="run_button"):
    st.session_state.simulation_complete = False # Reset flag
    # Define grid size here for consistency
    grid_w = 600
    grid_h = 600
    # Pass parameters to the updated model __init__
    st.session_state.model = DiningHallModel(N_agents, width=grid_w, height=grid_h,
                                           agent_speed=agent_speed,
                                           dining_wait_time=dining_wait_time,
                                           num_dining_spots=num_dining_spots,
                                           seed=random.randint(1, 10000)) # Add a random seed

    run_steps = 0
    # Run the simulation
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner(f"Running simulation..."):
        for i in range(max_steps):
            if st.session_state.model is None: # Should not happen, but safety check
                 st.error("Model disappeared during simulation run!")
                 break
            st.session_state.model.step()
            run_steps += 1

            # Update progress bar and status text periodically
            if i % 10 == 0: # Update every 10 steps
                progress = i / max_steps
                progress_bar.progress(progress)
                active_agents = len(st.session_state.model.agents) if st.session_state.model.agents else 0
                exited_agents = st.session_state.model.exited_count
                status_text.text(f"Step {run_steps}/{max_steps} | Active: {active_agents} | Exited: {exited_agents}/{N_agents}")

            # Updated stopping condition
            if st.session_state.model.exited_count >= N_agents and len(st.session_state.model.agents) == 0:
                 st.success(f"All {N_agents} agents exited by step {run_steps}.")
                 st.session_state.simulation_complete = True
                 progress_bar.progress(1.0)
                 status_text.text(f"Simulation Complete at step {run_steps}. All agents exited.")
                 break

        if not st.session_state.simulation_complete:
            active_agents = len(st.session_state.model.agents) if st.session_state.model.agents else 0
            final_message = (f"Simulation finished after {run_steps} steps (max steps reached). "
                             f"{st.session_state.model.exited_count}/{N_agents} agents exited. "
                             f"{active_agents} agents still active.")
            st.info(final_message)
            status_text.text(final_message)
            progress_bar.progress(1.0)
            st.session_state.simulation_complete = True


    # Store results in session state only after completion
    if st.session_state.model:
        st.session_state.results_df = st.session_state.model.datacollector.get_model_vars_dataframe()
    else:
        st.session_state.results_df = pd.DataFrame() # Ensure it exists but is empty


# --- Display Results ---
if st.session_state.model and st.session_state.simulation_complete:
    model = st.session_state.model

    st.header("Simulation Snapshot")
    st.write(f"Showing final state at step {model.steps}")

    # --- Matplotlib Visualization ---
    fig, ax = plt.subplots(figsize=(10, 10)) # Increased size slightly
    ax.set_xlim(0, model.grid_width)
    ax.set_ylim(0, model.grid_height)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Agent Positions with Background Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # --- Draw Background Map FIRST ---
    if st.session_state.uploaded_map_data is not None:
        try:
            extent = (map_left, map_right, map_bottom, map_top)
            ax.imshow(st.session_state.uploaded_map_data,
                      extent=extent,
                      alpha=map_alpha,
                      aspect='auto', # Let extent control aspect ratio within plot
                      origin='upper', # Often 'upper' works even with inverted y-axis later
                      zorder=0)      # Ensure it's behind everything else
        except Exception as e:
            st.warning(f"Could not display map image: {e}")


    # --- Draw Simulation Elements (Path, Spots, Agents) ---

    # Draw Path
    if show_path and hasattr(model, 'path_points') and model.path_points:
        path_x = [p[0] for p in model.path_points]
        path_y = [p[1] for p in model.path_points]
        ax.plot(path_x, path_y, color='lightgray', linestyle='--', linewidth=2, label="Path", zorder=1) # Increased zorder

    # Draw Dining Spots
    if show_dining_spots and hasattr(model, 'dining_spots'):
        for i, spot in enumerate(model.dining_spots):
            # Ensure spot has 'pos' and 'occupied_by'
            if isinstance(spot, dict) and "pos" in spot and "occupied_by" in spot:
                color = 'darkgreen' if spot["occupied_by"] is not None else 'lightgreen'
                spot_patch = patches.Circle(spot["pos"], radius=agent_radius_vis + 2, color=color, alpha=0.7, zorder=2) # Increased zorder
                ax.add_patch(spot_patch)
                ax.text(spot["pos"][0], spot["pos"][1], str(i), color='white', ha='center', va='center', fontsize=8, zorder=3) # Add spot number
            else:
                 st.warning(f"Dining spot data format incorrect: {spot}")


    # Draw Agents
    agent_colors = {
        "ENTERING": "gray", "MOVING": "blue", "APPROACHING_DINING": "orange",
        "MOVING_TO_SPOT": "purple", "WAITING": "red", "DINING": "pink",
        "REJOINING_PATH": "cyan", "EXITED": "black" # Should not be drawn if removed properly
    }

    # Access agents via model.agents (active agents)
    if hasattr(model, 'agents'):
        for agent in model.agents:
            if hasattr(agent, 'state') and hasattr(agent, 'pos'):
                color = agent_colors.get(agent.state, "black")
                # Make waiting agents slightly larger and red
                radius = agent_radius_vis * 1.5 if agent.state == "WAITING" else agent_radius_vis
                agent_patch = patches.Circle(agent.pos, radius=radius, color=color, alpha=0.9, zorder=4) # Highest zorder
                ax.add_patch(agent_patch)
            else:
                st.warning(f"Agent data format incorrect or incomplete for agent: {agent}")


    ax.invert_yaxis() # Invert Y axis AFTER plotting everything
    # Place legend outside plot area
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory


    # --- Data Analysis and Graphs ---
    st.header("Simulation Analysis")
    if 'results_df' in st.session_state and not st.session_state.results_df.empty:
        results_df = st.session_state.results_df

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Agent State Counts")
            plot_cols = ["Waiting", "Dining", "Moving", "TotalActiveAgents"]
            plot_cols = [col for col in plot_cols if col in results_df.columns]
            if plot_cols:
                st.line_chart(results_df[plot_cols])
            else:
                 st.warning("No data columns found for state counts chart.")

        with col2:
            st.subheader("Exited Agents Over Time")
            if "Exited" in results_df.columns:
                 st.line_chart(results_df[["Exited"]])
            else:
                st.warning("No 'Exited' column found.")

        st.subheader("Final Model Data")
        st.dataframe(results_df.tail())

    else:
        st.warning("No results data frame found in session state or simulation not run/completed.")

elif not st.session_state.model:
    st.info("Configure parameters in the sidebar and click 'Run Simulation'.")

# Add a message if the simulation is running but not complete
if st.session_state.get('model') and not st.session_state.get('simulation_complete'):
     st.info("Simulation running or finished prematurely. Check status messages above.")