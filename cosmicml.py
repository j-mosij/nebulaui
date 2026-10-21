import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CosmicNCAModule(nn.Module):
    """
    A modular local AI Neural Cellular Automata engine simulating 
    Fibonacci spiral gravitational sinks combined with Eagle Nebula gas growth.
    """
    def __init__(self, channel_dim=16, hidden_dim=32):
        super(CosmicNCAModule, self).__init__()
        self.channel_dim = channel_dim
        
        # Sobel filters to calculate local spatial derivatives (Perception Step)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t()
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        
        # Stack perception filters: Identity, grad_x, grad_y -> Input dimension becomes channel_dim * 3
        self.register_buffer('filters', torch.stack([identity, sobel_x, sobel_y]))
        
        # Local AI Update Rules (Simulating physics behavior via neural layers)
        self.w1 = nn.Conv2d(channel_dim * 3, hidden_dim, kernel_size=1)
        self.w2 = nn.Conv2d(hidden_dim, channel_dim, kernel_size=1)
        
        # Initialize final weights to near-zero to ensure stable initial growth
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def perceive(self, x):
        # Apply filters per channel to look at local neighbor states asynchronously
        b, c, h, w = x.shape
        filters = self.filters.unsqueeze(1).repeat(c, 1, 1, 1)
        perceived = F.conv2d(x, filters, groups=c, padding=1)
        # Reshape to group identity, dx, dy for each channel together
        perceived = perceived.view(b, c, 3, h, w).transpose(1, 2).reshape(b, c * 3, h, w)
        return perceived

    def forward(self, x, steps=1, fire_rate=0.5):
        for _ in range(steps):
            # 1. Perceive neighbors locally
            perceived = self.perceive(x)
            
            # 2. AI Network computes updates (Delta physics rules)
            dx = self.w1(perceived)
            dx = F.relu(dx)
            dx = self.w2(dx)
            
            # 3. Apply stochastic asynchronous cell mask (Not all cells fire at once)
            update_mask = torch.rand(x.size(0), 1, x.size(2), x.size(3), device=x.device) < fire_rate
            x = x + update_mask.float() * dx
            
            # 4. Enforce structural limits (e.g. RGB channels clipped between 0 and 1)
            x[:, :4, :, :] = torch.clamp(x[:, :4, :, :], 0.0, 1.0)
            
        return x

def seed_fibonacci_nebula(grid_size=128):
    """
    Creates an initial state seeding a Fibonacci spiral pattern 
    and nebula gas concentrations.
    """
    grid = torch.zeros(1, 16, grid_size, grid_size)
    center = grid_size // 2
    
    # Calculate golden ratio properties for local coordinate mapping
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi / (phi ** 2)
    
    for y in range(grid_size):
        for x in range(grid_size):
            dx, dy = x - center, y - center
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            if r > 0:
                # Fibonacci Spiral Vector Calculation
                # Logarithmic spiral equation: r = a * e^(b * theta)
                spiral_val = (np.log(r) / np.log(phi) - theta / golden_angle) % 1.0
                
                if r < 10:  # The Black Hole Singularity Core
                    grid[0, 0, y, x] = 0.1  # Low Red
                    grid[0, 1, y, x] = 0.0  # Zero Green
                    grid[0, 2, y, x] = 0.3  # Deep Blue Event Horizon
                    grid[0, 3, y, x] = 1.0  # Max Density Core
                    grid[0, 6, y, x] = 1.0  # Max Gravity Field
                elif spiral_val < 0.15:  # Nebula Dust/Gas Arms aligned to Fibonacci vectors
                    # Eagle Nebula colors: rich sulfur reds, hydrogen greens, oxygen blues
                    grid[0, 0, y, x] = np.clip(1.0 - (r / center), 0.2, 0.9)  # Red emission gas
                    grid[0, 1, y, x] = np.clip(spiral_val * 4, 0.1, 0.6)       # Green dust pillars
                    grid[0, 2, y, x] = np.clip(r / center, 0.3, 1.0)          # Outer cold ionization blue
                    grid[0, 3, y, x] = 0.8                                    # Matter concentration
                    
                    # Local directional velocities pulling inward & twisting clockwise
                    grid[0, 4, y, x] = (-dy / r) * 0.5 - (dx / r) * 0.2        # Vx
                    grid[0, 5, y, x] = (dx / r) * 0.5 - (dy / r) * 0.2         # Vy
                    grid[0, 6, y, x] = 1.0 / (r + 1)                          # Gravity Falloff
                    
    # Fill remaining hidden features with minor noise to bootstrap AI creativity
    grid[0, 7:, :, :] = torch.randn(1, 9, grid_size, grid_size) * 0.05
    return grid

# --- Initialization Example ---
if __name__ == "__main__":
    # Create the modular local CA model
    nca_model = CosmicNCAModule(channel_dim=16, hidden_dim=32)
    
    # Initialize the custom structural space seed
    universe_grid = seed_fibonacci_nebula(grid_size=128)
    
    print(f"Initial Universe State Created: {universe_grid.shape}")
    print("Simulating 5 steps of localized modular physics...")
    
    # Run simulation iterations
    updated_universe = nca_model(universe_grid, steps=5)
    print(f"Simulation completed. Output state shape: {updated_universe.shape}")
