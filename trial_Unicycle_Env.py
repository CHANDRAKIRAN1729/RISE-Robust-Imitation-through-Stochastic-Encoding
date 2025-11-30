# Import Gymnasium for creating RL environments
import gymnasium as gym
# Import spaces module to define action and observation spaces
from gymnasium import spaces
# Import numpy for numerical operations
import numpy as np
# Import pygame for rendering visualization
import pygame
# Import os for file path operations
import os

class TrialUnicycleEnv(gym.Env):
    """Unicycle environment with one moving circular obstacle and a goal.

    Observation: np.array([x, y, theta], dtype=float32)
    Action: np.array([v, w], dtype=float32) with v,w in [-1,1]
    Terminated when either reaching goal or colliding with obstacle.
    """
    # Set rendering frame rate to 30 FPS
    metadata = {"render_fps": 30}

    def __init__(self):
        # Call parent Gymnasium environment constructor
        super(TrialUnicycleEnv, self).__init__()
        # Initialize pygame screen as None (created on first render)
        self.screen = None
        
        # Define action space: 2D continuous actions [linear_vel, angular_vel] both in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Define observation space: 3D state [x, y, theta] where x∈[-3,3], y∈[-1,1], theta∈[-π,π]
        self.observation_space = spaces.Box(
            low=np.array([-3, -1, -np.pi], dtype=np.float32),
            high=np.array([3, 1, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Define overall playground boundaries (6×2 rectangular area)
        self.rect_bounds = {'xmin': -3, 'xmax': 3, 'ymin': -1, 'ymax': 1}
        # Define agent spawn zone on left side (prevents same starting position every episode)
        self.agent_spawn_zone = {'xmin': -2.8, 'xmax': -1.5, 'ymin': -0.8, 'ymax': 0.8}
        # Define goal spawn zone on right side
        self.goal_zone = {'xmin': 2, 'xmax': 3, 'ymin': -1, 'ymax': 1}
        # Define obstacle movement zone in center (obstacle stays here)
        self.obstacle_zone = {'xmin': -2, 'xmax': 2, 'ymin': -1, 'ymax': 1}
        
        # Define obstacle velocity range (each x,y component can be [-0.3, 0.3] m/s)
        self.obstacle_velocity_range = {'low': -0.3, 'high': 0.3}

        # Define agent physical radius (0.07m) for collision detection
        self.agent_radius = 0.07

        # Initialize crash flag to False
        self.crashed = False
        # Initialize step counter
        self.counter = 0
        # Initialize goal reach counter (across all episodes)
        self.reach_completion_count = 0
        # Initialize collision counter (across all episodes)
        self.obstacle_violation_count = 0

        # Initialize random number generator (will be set by reset with seed)
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # set numpy generator for reproducibility
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        elif self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random()

        self.counter = 0
        self.crashed = False

        # Spawn agent in its designated spawn zone (left end)
        agent_x = self.np_random.uniform(self.agent_spawn_zone['xmin'], self.agent_spawn_zone['xmax'])
        agent_y = self.np_random.uniform(self.agent_spawn_zone['ymin'], self.agent_spawn_zone['ymax'])
        agent_theta = self.np_random.uniform(-np.pi, np.pi)
        self.state = np.array([agent_x, agent_y, agent_theta], dtype=np.float32)

        # Spawn goal in the goal zone (right end)
        goal_x = self.np_random.uniform(self.goal_zone['xmin'], self.goal_zone['xmax'])
        goal_y = self.np_random.uniform(self.goal_zone['ymin'], self.goal_zone['ymax'])
        self.goal = np.array([goal_x, goal_y], dtype=np.float32)

        # Place the obstacle randomly within the obstacle zone with simple clearance checks
        valid_obstacle = False
        while not valid_obstacle:
            obs_x = self.np_random.uniform(self.obstacle_zone['xmin'], self.obstacle_zone['xmax'])
            obs_y = self.np_random.uniform(self.obstacle_zone['ymin'], self.obstacle_zone['ymax'])
            self.obstacle_position = np.array([obs_x, obs_y], dtype=np.float32)
            if (np.linalg.norm(self.obstacle_position - self.state[:2]) > 0.8 and
                np.linalg.norm(self.obstacle_position - self.goal) > 1.0):
                valid_obstacle = True

        # Pick a random constant velocity for the obstacle from the given range
        vx = self.np_random.uniform(self.obstacle_velocity_range['low'], self.obstacle_velocity_range['high'])
        vy = self.np_random.uniform(self.obstacle_velocity_range['low'], self.obstacle_velocity_range['high'])
        self.obstacle_velocity = np.array([vx, vy], dtype=np.float32)

        # Randomize obstacle radius a bit
        self.obstacle_radius = float(self.np_random.uniform(0.08, 0.18))

        return self.state.copy(), {}
    
    def step(self, action):
        # Extract current x, y, theta from state
        x, y, theta = self.state
        # Clip action to valid range [-1,1] for safety
        v_new, w_new = np.clip(action, self.action_space.low, self.action_space.high)
        # Set time step to 0.1 seconds (10 Hz update rate)
        dt = 0.1
        
        # Update heading angle using angular velocity (unicycle kinematics)
        theta += w_new * dt
        # Update x position using linear velocity and current heading
        x += v_new * np.cos(theta) * dt
        # Update y position using linear velocity and current heading
        y += v_new * np.sin(theta) * dt
        
        # Constrain x to stay within playground bounds
        x = np.clip(x, self.rect_bounds['xmin'], self.rect_bounds['xmax'])
        # Constrain y to stay within playground bounds
        y = np.clip(y, self.rect_bounds['ymin'], self.rect_bounds['ymax'])
        # Update state with new position and orientation
        self.state = np.array([x, y, theta], dtype=np.float32)
        # Increment step counter
        self.counter += 1
        
        # Calculate where obstacle would move with current velocity
        new_obs_pos = self.obstacle_position + self.obstacle_velocity * dt
        
        # Check if obstacle would exit zone horizontally, if so reverse x-velocity (bounce)
        if new_obs_pos[0] < self.obstacle_zone['xmin'] or new_obs_pos[0] > self.obstacle_zone['xmax']:
            self.obstacle_velocity[0] *= -1
        # Check if obstacle would exit zone vertically, if so reverse y-velocity (bounce)
        if new_obs_pos[1] < self.obstacle_zone['ymin'] or new_obs_pos[1] > self.obstacle_zone['ymax']:
            self.obstacle_velocity[1] *= -1
        
        # Move obstacle by velocity × time (after possible bounce correction)
        self.obstacle_position += self.obstacle_velocity * dt
        
        # Calculate Euclidean distance from agent to goal
        distance_to_goal = float(np.linalg.norm(self.state[:2] - self.goal))
        # Negative distance reward (encourages getting closer to goal)
        reward = -distance_to_goal

        # Calculate distance between agent center and obstacle center
        obs_dist = float(np.linalg.norm(self.state[:2] - self.obstacle_position))
        # Collision occurs when centers are closer than sum of radii
        collision = obs_dist <= (self.obstacle_radius + self.agent_radius)
        # Penalize collision with -10 reward
        if collision:
            reward -= 10.0
            # Increment total collision count
            self.obstacle_violation_count += 1

        # Check if agent reached goal (within 0.2m)
        reached = distance_to_goal < 0.2
        # Reward reaching goal with +10
        if reached:
            reward += 10.0
            # Increment total goal reach count
            self.reach_completion_count += 1

        # Episode terminates if collision OR goal reached
        terminated = bool(collision or reached)
        # No time-based truncation (controlled externally)
        truncated = False

        # Package additional information for logging/debugging
        info = {
            'reached_goal': reached,
            'collision': collision,
            'goal': self.goal.copy(),
            'obstacle_position': self.obstacle_position.copy(),
            'obstacle_velocity': self.obstacle_velocity.copy(),
            'obstacle_radius': float(self.obstacle_radius),
            'agent_radius': float(self.agent_radius),
        }
        # Return new state, reward, termination flags, and info
        return self.state.copy(), reward, terminated, truncated, info


    def render(self):
        # Check if pygame window not created yet
        if self.screen is None:
            # Initialize pygame library
            pygame.init()
            # Create 1000x500 pixel window
            self.screen = pygame.display.set_mode((1000, 500))
            # Set window title
            pygame.display.set_caption("Unicycle Environment")
            # Create clock for FPS control
            self.clock = pygame.time.Clock()
            # Get directory where this script is located
            assets_dir = os.path.dirname(__file__)
            # Build path to car image
            car_path = os.path.join(assets_dir, "red-top-car.png")
            # Build path to flag image
            flag_path = os.path.join(assets_dir, "flag.png")
            # Try loading car image
            try:
                # Load car image from file
                self.car_img = pygame.image.load(car_path)
                # Resize to 40x40 pixels
                self.car_img = pygame.transform.scale(self.car_img, (40, 40))
            # If image not found, create fallback
            except Exception:
                # Create transparent 40x40 surface
                self.car_img = pygame.Surface((40, 40), pygame.SRCALPHA)
                # Draw blue triangle as fallback car
                pygame.draw.polygon(self.car_img, (0, 0, 255), [(20, 0), (40, 40), (0, 40)])
            # Try loading flag image
            try:
                # Load flag image from file
                self.flag_img = pygame.image.load(flag_path)
                # Resize to 50x30 pixels
                self.flag_img = pygame.transform.scale(self.flag_img, (50, 30))
            # If image not found, create fallback
            except Exception:
                # Create transparent 50x30 surface
                self.flag_img = pygame.Surface((50, 30), pygame.SRCALPHA)
                # Draw green rectangle as fallback flag
                pygame.draw.rect(self.flag_img, (0, 128, 0), self.flag_img.get_rect())
        
        # Fill screen with white background
        self.screen.fill((255, 255, 255))
        
        # Define helper to convert env coords [-3,3]×[-1,1] to screen [0,1000]×[0,500]
        def to_screen_coords(x, y):
            # Map x: center at 500, scale by 150
            screen_x = int(500 + x * 150)
            # Map y: center at 250, flip (screen y grows down)
            screen_y = int(250 - y * 150)
            # Return pixel coordinates
            return screen_x, screen_y

        # Calculate left edge of agent zone in pixels
        agent_left = int(500 + self.agent_spawn_zone['xmin'] * 150)
        # Calculate top edge of agent zone
        agent_top = int(250 - self.agent_spawn_zone['ymax'] * 150)
        # Calculate width of agent zone in pixels
        agent_width = int((self.agent_spawn_zone['xmax'] - self.agent_spawn_zone['xmin']) * 150)
        # Calculate height of agent zone in pixels
        agent_height = int((self.agent_spawn_zone['ymax'] - self.agent_spawn_zone['ymin']) * 150)
        # Check if zone has valid dimensions
        if agent_width > 0 and agent_height > 0:
            # Create translucent surface for agent zone
            agent_surf = pygame.Surface((agent_width, agent_height), pygame.SRCALPHA)
            # Fill with white color, alpha=80 (semi-transparent)
            agent_surf.fill((255, 255, 255, 80))
            # Draw zone on screen at calculated position
            self.screen.blit(agent_surf, (agent_left, agent_top))
        
        # Calculate left edge of obstacle zone
        obst_left = int(500 + self.obstacle_zone['xmin'] * 150)
        # Calculate top edge of obstacle zone
        obst_top = int(250 - self.obstacle_zone['ymax'] * 150)
        # Calculate width of obstacle zone
        obst_width = int((self.obstacle_zone['xmax'] - self.obstacle_zone['xmin']) * 150)
        # Calculate height of obstacle zone
        obst_height = int((self.obstacle_zone['ymax'] - self.obstacle_zone['ymin']) * 150)
        # Check if zone has valid dimensions
        if obst_width > 0 and obst_height > 0:
            # Create translucent surface for obstacle zone
            obst_surf = pygame.Surface((obst_width, obst_height), pygame.SRCALPHA)
            # Fill with white, alpha=80
            obst_surf.fill((255, 255, 255, 80))
            # Draw zone on screen
            self.screen.blit(obst_surf, (obst_left, obst_top))
        
        # Calculate left edge of goal zone
        goal_left = int(500 + self.goal_zone['xmin'] * 150)
        # Calculate top edge of goal zone
        goal_top = int(250 - self.goal_zone['ymax'] * 150)
        # Calculate width of goal zone
        goal_width = int((self.goal_zone['xmax'] - self.goal_zone['xmin']) * 150)
        # Calculate height of goal zone
        goal_height = int((self.goal_zone['ymax'] - self.goal_zone['ymin']) * 150)
        # Check if zone has valid dimensions
        if goal_width > 0 and goal_height > 0:
            # Create translucent surface for goal zone
            goal_surf = pygame.Surface((goal_width, goal_height), pygame.SRCALPHA)
            # Fill with white, alpha=80
            goal_surf.fill((255, 255, 255, 80))
            # Draw zone on screen
            self.screen.blit(goal_surf, (goal_left, goal_top))
        
        # Convert obstacle position to screen coordinates
        obst_screen_pos = to_screen_coords(self.obstacle_position[0], self.obstacle_position[1])
        # Draw red circle for obstacle
        pygame.draw.circle(
            self.screen, (255, 0, 0),  # Red color
            obst_screen_pos,  # Center position
            int(self.obstacle_radius * 150)  # Radius in pixels
        )
        
        # Get rectangle for flag centered at goal position
        flag_rect = self.flag_img.get_rect(center=to_screen_coords(self.goal[0], self.goal[1]))
        # Draw flag image at goal location
        self.screen.blit(self.flag_img, flag_rect.topleft)
        
        # Rotate car image by agent's heading (+180 to fix orientation)
        rotated_car = pygame.transform.rotate(self.car_img, np.degrees(self.state[2]) + 180)
        # Get rectangle for car centered at agent position
        car_rect = rotated_car.get_rect(center=to_screen_coords(self.state[0], self.state[1]))
        # Draw rotated car at agent location
        self.screen.blit(rotated_car, car_rect.topleft)

        # Update display to show all drawings
        pygame.display.flip()
        # Limit to 30 FPS as specified in metadata
        self.clock.tick(self.metadata.get("render_fps", 30))
    
    def close(self):
        # Properly shut down pygame and release resources
        pygame.quit()

# Example of running the environment
if __name__ == '__main__':
    # Create environment instance
    env = TrialUnicycleEnv()
    # Reset environment to initial state
    obs = env.reset()
    # Initialize done flag
    done = False
    # Initialize terminated flag (goal/collision)
    terminated = False
    # Initialize truncated flag (time limit)
    truncated = False
    # Run until episode ends
    while not (terminated or truncated):
        # Render visualization
        env.render()
        # Sample random action from action space
        action = env.action_space.sample()
        # Execute action and get results
        obs, reward, terminated, truncated, info = env.step(action)
    # Clean up resources
    env.close()