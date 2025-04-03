import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mesa
import numpy as np
import random
from model import DiningHallModel, MonkAgent

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Monk Crowd Simulation")

st.title("Monastery Dining Hall Simulation (Agent-Based Model - Mesa 3.x)")
st.write("Simulating monks moving along a path, stopping at dining spots, and exiting.")

# --- Simulation Parameters (Sidebar) ---
st.sidebar.header("Simulation Parameters")
N_agents = st.sidebar.slider("Number of Monks (Target)", 5, 500, 100)
agent_speed = st.sidebar.slider("Average Agent Speed", 1.0, 5.0, 2.0, 0.1)
dining_wait_time = st.sidebar.slider("Dining Wait Time (steps)", 10, 200, 50)
num_dining_spots = st.sidebar.slider("Number of Dining Spots", 1, 10, 10)
max_steps = st.sidebar.number_input("Maximum Simulation Steps", 100, 5000, 1000)

# --- Visualization Settings ---
st.sidebar.header("Visualization")
show_path = st.sidebar.checkbox("Show Path", True)
show_dining_spots = st.sidebar.checkbox("Show Dining Spots", True)
agent_radius_vis = 3 # For visualization scale

# --- Simulation Execution ---
# Use session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'simulation_complete' not in st.session_state:
    st.session_state.simulation_complete = False

# Button to start/restart simulation
if st.sidebar.button("Run Simulation"):
    st.session_state.simulation_complete = False # Reset flag
    # Pass parameters to the updated model __init__
    st.session_state.model = DiningHallModel(N_agents, width=600, height=600,
                                           agent_speed=agent_speed,
                                           dining_wait_time=dining_wait_time,
                                           num_dining_spots=num_dining_spots,
                                           seed=random.randint(1, 10000)) # Add a random seed
    
    run_steps = 0
    # Run the simulation
    with st.spinner(f"Running simulation..."):
        for i in range(max_steps):
            st.session_state.model.step()
            run_steps += 1
            # Updated stopping condition: Check if exited count >= target
            # AND no more active agents are left (important if some get stuck)
            if st.session_state.model.exited_count >= N_agents and len(st.session_state.model.agents) == 0:
                 st.success(f"All {N_agents} agents exited by step {run_steps}.")
                 st.session_state.simulation_complete = True
                 break
        
        if not st.session_state.simulation_complete:
            st.info(f"Simulation finished after {run_steps} steps. "
                  f"{st.session_state.model.exited_count}/{N_agents} agents exited. "
                  f"{len(st.session_state.model.agents)} agents still active.")
            st.session_state.simulation_complete = True


    # Store results in session state only after completion
    st.session_state.results_df = st.session_state.model.datacollector.get_model_vars_dataframe()
    # Optional: Get agent data (can be large!)
    # try:
    #      st.session_state.agent_results_df = st.session_state.model.datacollector.get_agent_vars_dataframe()
    # except Exception as e:
    #      st.warning(f"Could not retrieve agent-level data: {e}")
    #      st.session_state.agent_results_df = pd.DataFrame()


# --- Display Results ---
if st.session_state.model and st.session_state.simulation_complete:
    model = st.session_state.model

    st.header("Simulation Snapshot")
    # Use model.steps (the automatic counter)
    st.write(f"Showing final state at step {model.steps}")

    # --- Matplotlib Visualization ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, model.grid_width)
    ax.set_ylim(0, model.grid_height)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Agent Positions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Draw Path (Unchanged)
    if show_path and model.path_points:
        path_x = [p[0] for p in model.path_points]
        path_y = [p[1] for p in model.path_points]
        ax.plot(path_x, path_y, color='lightgray', linestyle='--', linewidth=1, label="Path")

    # Draw Dining Spots (Unchanged visually, relies on model.dining_spots)
    if show_dining_spots:
        for i, spot in enumerate(model.dining_spots):
            color = 'darkgreen' if spot["occupied_by"] is not None else 'lightgreen'
            spot_patch = patches.Circle(spot["pos"], radius=agent_radius_vis + 1, color=color, alpha=0.6)
            ax.add_patch(spot_patch)

    # Draw Agents
    agent_colors = { # Use same states
        "ENTERING": "gray", "MOVING": "blue", "APPROACHING_DINING": "orange",
        "MOVING_TO_SPOT": "purple", "WAITING": "red", "DINING": "pink",
        "REJOINING_PATH": "cyan", "EXITED": "black" # Should not be drawn if removed properly
    }
    
    # Access agents via model.agents
    # Note: model.agents contains only *active* agents. Exited agents removed via agent.remove() won't be here.
    for agent in model.agents:
            color = agent_colors.get(agent.state, "black")
            radius = agent_radius_vis * 1.5 if agent.state == "WAITING" else agent_radius_vis
            agent_patch = patches.Circle(agent.pos, radius=radius, color=color, alpha=0.9)
            ax.add_patch(agent_patch)

    ax.invert_yaxis()
    # Place legend outside plot area if needed
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout() # Adjust layout
    st.pyplot(fig)
    plt.close(fig)


    # --- Data Analysis and Graphs ---
    st.header("Simulation Analysis")
    if 'results_df' in st.session_state and not st.session_state.results_df.empty:
        results_df = st.session_state.results_df

        st.subheader("Agent State Counts Over Time")
        # Make sure column names match DataCollector keys ("Moving" might need adjustment if split)
        # Ensure the 'Moving' key in DataCollector aggregates all moving-like states if needed for the chart
        # The updated model.py combines moving states for the collector key "Moving"
        plot_cols = ["Waiting", "Dining", "Moving", "TotalActiveAgents"]
        # Filter out columns that might not exist if DC failed partially
        plot_cols = [col for col in plot_cols if col in results_df.columns]
        if plot_cols:
            st.line_chart(results_df[plot_cols])
        else:
             st.warning("No data columns found for state counts chart.")


        st.subheader("Exited Agents Over Time")
        if "Exited" in results_df.columns:
             st.line_chart(results_df[["Exited"]])
        else:
            st.warning("No 'Exited' column found.")


        st.subheader("Final Model Data")
        st.dataframe(results_df.tail())

        # Display agent data if collected
        # if 'agent_results_df' in st.session_state and not st.session_state.agent_results_df.empty:
        #     st.subheader("Agent-Level Data (Sample)")
        #     st.dataframe(st.session_state.agent_results_df.head()) # Show first few rows

    else:
        st.warning("No results data frame found in session state.")

elif not st.session_state.model:
    st.info("Configure parameters in the sidebar and click 'Run Simulation'.")

# Add a message if the simulation is running
if st.session_state.get('model') and not st.session_state.get('simulation_complete'):
     st.info("Simulation is running or has finished without displaying results yet. Check sidebar/terminal for errors if stuck.")