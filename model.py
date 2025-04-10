import mesa
import numpy as np
import random

# --- Helper Functions (Unchanged) ---

def distance(pos1, pos2):
  """Calculate Euclidean distance."""
  return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def move_towards(current_pos, target_pos, speed):
  """Calculate the new position after moving towards the target."""
  current_pos = np.array(current_pos)
  target_pos = np.array(target_pos)
  direction = target_pos - current_pos
  dist = np.linalg.norm(direction)

  if dist == 0 or dist < speed:
      return tuple(target_pos) # Reached target

  norm_direction = direction / dist
  new_pos = current_pos + norm_direction * speed
  return tuple(new_pos)

# --- Agent Definition (Mesa 3.x) ---

class MonkAgent(mesa.Agent):
  """Represents a monk navigating the path and dining."""

  # Note: unique_id is REMOVED from __init__ parameters
  def __init__(self, model, speed_variation=0.1, agent_radius=3):
    # Note: unique_id is REMOVED from super() call
    super().__init__(model) # Mesa assigns unique_id automatically
    self.state = "ENTERING"
    self.pos = self.model.path_points[0]
    self.path_index = 0
    self.base_speed = self.model.agent_speed
    self.speed = self.base_speed + random.uniform(-speed_variation, speed_variation) * self.base_speed
    self.wait_timer = 0
    self.assigned_spot_index = None
    self.target_pos = None
    self.agent_radius = agent_radius
    self.approach_timer = 0  # Add a timer for approaching state
    self.has_dined = False   # Ensure dining happens only once

  def is_agent_ahead_too_close(self):
    """Check if the agent directly ahead (ID-1) is too close."""
    # Check requires knowing the agent's *actual* ID after Mesa assigns it
    # This logic might need rethinking in Mesa 3 - checking *all nearby* agents might be better
    # For now, let's adapt using the model's agent list:
    
    # Get all active agents except self
    other_agents = [a for a in self.model.agents if a.unique_id != self.unique_id]
    
    required_spacing = self.agent_radius * 2.5 # Desired minimum distance
    
    # Check agents near our current position
    for agent_ahead in other_agents:
         # Only consider agents roughly 'ahead' on the same path segment or state
         on_similar_segment = (self.path_index > 0 and self.path_index == agent_ahead.path_index) or \
                                (self.state == agent_ahead.state and self.state in ["APPROACHING_DINING", "REJOINING_PATH", "MOVING_TO_SPOT"])

         if on_similar_segment:
              dist_ahead = distance(self.pos, agent_ahead.pos)
              if dist_ahead < required_spacing:
                   # Simplified check: Is the other agent closer to the current target point than us?
                   # This is an approximation of being "ahead".
                   my_dist_to_target = distance(self.pos, self.target_pos) if self.target_pos else float('inf')
                   their_dist_to_target = distance(agent_ahead.pos, self.target_pos) if self.target_pos else float('inf')

                   if their_dist_to_target < my_dist_to_target :
                        return True # An agent is ahead and too close

    return False # No one problematic found

  def avoid_dining_spots(self, next_pos):
    """Check if the next position would collide with any dining spot that's not ours."""
    safe_dist = self.agent_radius * 2
    
    for i, spot in enumerate(self.model.dining_spots):
        if i != self.assigned_spot_index and spot["occupied_by"] is not None:
            if distance(next_pos, spot["pos"]) < safe_dist:
                return True  # Too close to someone else's dining spot
    return False

  def step(self):
    """Execute one step of the agent's logic."""

    if self.state == "EXITED":
      return # Agent already removed logic below

    # --- Basic Spacing Logic ---
    # Use updated check
    if self.target_pos and self.is_agent_ahead_too_close():
       # If too close to an agent ahead, pause this step
       return

    tol = 1e-2  # tolerance for position comparisons

    # --- State Machine Logic ---
    if self.state == "ENTERING":
        self.path_index = 1
        if self.path_index < len(self.model.path_points):
             self.target_pos = self.model.path_points[self.path_index]
             self.state = "MOVING"
        else: # Path has less than 2 points? Exit immediately.
             self.state = "EXITED"
             self.model.exited_count += 1
             self.remove() # Use agent.remove()
             return

    elif self.state == "MOVING":
      if distance(self.pos, self.target_pos) < tol:
          if self.path_index == self.model.dining_entry_path_index:
              self.state = "APPROACHING_DINING"
              self.approach_timer = 0  # Reset approach timer
          elif self.path_index >= len(self.model.path_points) - 1:
              self.state = "EXITED"
              self.model.exited_count += 1
              self.remove() # Use agent.remove()
              return
          else:
              self.path_index += 1
              self.target_pos = self.model.path_points[self.path_index]

      # Move towards the current target path point only if target exists
      if self.target_pos:
            # Calculate next position
            next_pos = move_towards(self.pos, self.target_pos, self.speed)
            self.pos = next_pos

    elif self.state == "APPROACHING_DINING":
       # Add a small delay to stagger agents
       self.approach_timer += 1
       if self.approach_timer < self.unique_id % 3:  # Use ID to stagger approaches
           return
           
       spot_index = self.model.find_available_dining_spot(self.unique_id, self.pos)
       if spot_index is not None:
          self.assigned_spot_index = spot_index
          self.target_pos = self.model.dining_spots[spot_index]["pos"]
          self.state = "MOVING_TO_SPOT"
       else:
          # Remain at the dining entry point waiting for a free spot
          self.pos = self.model.path_points[self.model.dining_entry_path_index]

    elif self.state == "MOVING_TO_SPOT":
      if self.target_pos:
        if distance(self.pos, self.target_pos) < tol:
          self.state = "DINING"
          self.wait_timer = 0  # start dining timer
        else:
          self.pos = move_towards(self.pos, self.target_pos, self.speed)

    elif self.state == "WAITING":
       self.wait_timer += 1
       if self.wait_timer >= self.model.dining_wait_time:
           self.state = "DINING"

    elif self.state == "DINING":
        if not self.has_dined:
            self.wait_timer += 1
            if self.wait_timer >= self.model.dining_wait_time:
                self.has_dined = True
                # Release the dining spot once dinner is complete
                if self.assigned_spot_index is not None:
                    self.model.release_dining_spot(self.assigned_spot_index)
                    self.assigned_spot_index = None
                self.state = "REJOINING_PATH"
                self.path_index = self.model.dining_exit_path_index
                if self.path_index < len(self.model.path_points):
                    self.target_pos = self.model.path_points[self.path_index]
                else:
                    self.state = "EXITED"
                    self.model.exited_count += 1
                    self.remove()
                    return
        else:
            # Safety: if already dined, ensure agent remains dining for its full duration before exiting
            self.state = "REJOINING_PATH"
            self.path_index = self.model.dining_exit_path_index
            self.target_pos = self.model.path_points[self.path_index]

    elif self.state == "REJOINING_PATH":
      if self.target_pos:
          if distance(self.pos, self.target_pos) < tol:
               self.state = "MOVING"
               if self.path_index < len(self.model.path_points) - 1:
                    self.path_index += 1
                    self.target_pos = self.model.path_points[self.path_index]
               else:
                   self.state = "EXITED"
                   self.model.exited_count += 1
                   self.remove() # Use agent.remove()
                   return
          else:
              # Calculate next position while avoiding dining spots
              next_pos = move_towards(self.pos, self.target_pos, self.speed)
              
              # If we're rejoining the path, avoid going through dining spots
              if self.avoid_dining_spots(next_pos):
                  # Go around by taking a direct path to the exit point
                  exit_point = self.model.path_points[self.model.dining_exit_path_index]
                  next_pos = move_towards(self.pos, exit_point, self.speed * 0.8)
                  
              self.pos = next_pos


# --- Model Definition (Mesa 3.x) ---

class DiningHallModel(mesa.Model):
  """Manages the simulation environment and agents."""

  def __init__(self, N_agents, width, height, agent_speed, dining_wait_time, num_dining_spots=10, seed=None):
    # Mandatory super().__init__() - pass seed here if provided
    super().__init__(seed=seed)
    self.num_agents_target = N_agents # Renamed for clarity
    self.grid_width = width
    self.grid_height = height
    self.agent_speed = agent_speed
    self.dining_wait_time = dining_wait_time
    self.num_dining_spots = num_dining_spots
    
    # Agent Creation Tracking / Control
    self.created_agent_count = 0
    # Stagger creation: attempt every N steps (can be adjusted)
    self.agent_creation_interval = 5 # Try to add an agent every 5 steps initially

    # Path definition (remains the same logic)
    self.path_points = [
        (width, 180),         # 0: Start
        (110, 180),           # 1: Turn down
        (110, 60),           # 2: Arrive dining latitude
        (75, 60),           # 3: *Dining Entry Point*
        (75, 340),           # 4: *Dining Exit Point*
        (180, 340),           # 5: Turn away (moved further from dining spots)
        (180, height + 20),   # 6: Exit point
    ]
    self.dining_entry_path_index = 3
    self.dining_exit_path_index = 4

    # Dining spots calculation (remains the same logic)
    self.dining_spots = []
    dining_x = 65  # Move dining spots further left away from the path
    if self.dining_entry_path_index < len(self.path_points) and self.dining_exit_path_index < len(self.path_points):
        path_y_start = 85
        path_y_end = 305
        spot_spacing = (path_y_end - path_y_start) / (self.num_dining_spots + 1) if self.num_dining_spots > 0 else 0
        for i in range(self.num_dining_spots):
            spot_y = path_y_start + (i + 1) * spot_spacing
            self.dining_spots.append({"pos": (dining_x, spot_y), "occupied_by": None})
    else:
        print("Warning: Invalid dining entry/exit path indices. No dining spots created.")


    # REMOVED: self.schedule = ... No schedulers in Mesa 3.x!
    self.exited_count = 0

    # Data Collector - updated agent access and counting
    # Directly instantiate DataCollector
    self.datacollector = mesa.DataCollector(
        model_reporters={
            # Use model.agents directly
            "Waiting": lambda m: sum(1 for a in m.agents if a.state == "WAITING"),
            "Dining": lambda m: sum(1 for a in m.agents if a.state == "DINING"),
            "Moving": lambda m: sum(1 for a in m.agents if a.state == "MOVING" or a.state=="APPROACHING_DINING" or a.state=="MOVING_TO_SPOT" or a.state=="REJOINING_PATH"), # Combine moving states
            "Exited": lambda m: m.exited_count,
            # Use len(model.agents) for count
            "TotalActiveAgents": lambda m: len(m.agents)
        },
        # Agent reporters are usually fine unless you used scheduler-specific agent properties
        agent_reporters={"State": "state", "Position": "pos"}
    )

  def find_available_dining_spot(self, agent_id, agent_pos):
      """Finds the closest available spot and marks it as occupied."""
      candidate_index = None
      min_dist = float('inf')
      for i, spot in enumerate(self.dining_spots):
          if spot["occupied_by"] is None:
              d = distance(agent_pos, spot["pos"])
              if d < min_dist:
                  min_dist = d
                  candidate_index = i
      if candidate_index is not None:
          self.dining_spots[candidate_index]["occupied_by"] = agent_id
      return candidate_index

  def release_dining_spot(self, spot_index):
      """Marks a dining spot as free."""
      if 0 <= spot_index < len(self.dining_spots):
           self.dining_spots[spot_index]["occupied_by"] = None

  def try_add_agent(self):
      """Attempts to add a new agent if conditions are met."""
      if self.created_agent_count >= self.num_agents_target:
          return # Don't add more than requested

      # Simple stagger: try only every N steps (adjust interval as needed)
      # More complex logic could involve checking density near entry
      if self.steps % self.agent_creation_interval != 0:
           return

      # Check if entry point is clear enough
      entry_pos = self.path_points[0]
      clear_to_enter = True
      # Estimate radius without creating a full temp agent if possible
      temp_agent_radius = 3 # Assume default radius for check
      min_entry_dist = temp_agent_radius * 4

      # Access agents via model.agents
      for agent in self.agents:
           # Only consider agents still near the entrance
           if agent.state in ["ENTERING", "MOVING"] and agent.path_index <= 1:
                if distance(agent.pos, entry_pos) < min_entry_dist:
                    clear_to_enter = False
                    break

      if clear_to_enter:
          # Just create the agent, Mesa adds it automatically. No unique_id needed.
          agent = MonkAgent(self)
          self.created_agent_count += 1
          # Optional: Slow down creation rate once many agents exist
          # if self.created_agent_count > self.num_agents_target / 2:
          #      self.agent_creation_interval = 10 # Example adjustment
          #print(f"Step {self.steps}: Added agent {agent.unique_id}") # unique_id is assigned now

      # After adding many agents, slow down the creation rate
      if self.created_agent_count > self.num_agents_target / 2:
          self.agent_creation_interval = max(3, self.agent_creation_interval)  # Adjust minimum interval

  def step(self):
    """Advance the model by one step."""
    self.try_add_agent()        # Try adding agents first

    # Execute step() for all agents in random order
    # This REPLACES self.schedule.step() for RandomActivation
    self.agents.shuffle().do("step") # Shuffle returns a new shuffled AgentSet, then call do()

    self.datacollector.collect(self) # Collect data

    # NEW: Increment simulation steps manually for graph reporting
    self.steps = getattr(self, 'steps', 0) + 1