
## **RISE Project Methods Explained**

### **🎯 Layman's Terms**

Imagine you're teaching a robot to drive a car and avoid obstacles:

**1. Behavior Cloning (PC-BC):**
Think of this like a student driver who directly copies an experienced driver. You record an expert driver navigating through traffic, noting exactly what they do in each situation (speed, steering). Then you train the robot to mimic these actions when it sees similar situations. It's straightforward copying: "When you see this obstacle here moving this fast, turn the wheel this much."

**2. VAE Policy (RISE):**
This is like teaching the student driver to first *understand the concept of danger* before learning to drive. It works in two stages:
- **Stage 1**: The robot learns to recognize and compress obstacle information into a simple "danger code" (like understanding "fast-moving truck" vs "slow bicycle")
- **Stage 2**: The robot learns to drive using this danger code instead of raw obstacle details, making it better at handling new, slightly different obstacles it hasn't seen before

**The Key Difference:** BC is like rote memorization, while RISE learns underlying patterns first, making it more adaptable to new situations.

---

### **🔬 Technical Deep Dive**

#### **1. Behavior Cloning (PC-BC) - Parameter-Conditioned Baseline**

**Architecture:**
- Simple feedforward neural network (2-layer MLP with 128 hidden units)
- **Input**: Concatenation of:
  - Robot state (x, y, θ): 3D position and orientation
  - Goal position (gx, gy): 2D target
  - Obstacle parameters (ox, oy, vx, vy, radius): 5D raw obstacle info
- **Output**: Action (v, w): Linear and angular velocities

**Training Process:**
```
1. Collect expert demonstrations using oracle controller
2. Train network with supervised learning (MSE loss)
   Loss = ||predicted_action - expert_action||²
3. Direct mapping: π(s, g, c) → a
```

**Strengths:**
- Simple, interpretable architecture
- Fast training (single-stage)
- Low computational cost

**Weaknesses:**
- **Overfitting to specific obstacle configurations**: The network memorizes exact obstacle parameters from training data
- **Poor generalization**: Struggles when obstacles have slightly different sizes or velocities not seen during training
- **No uncertainty handling**: Treats all inputs deterministically

---

#### **2. VAE Policy (RISE) - Robust Imitation through Stochastic Encoding**

**Architecture Components:**

**A. Variational Autoencoder (VAE):**
- **Encoder**: Maps 5D obstacle params → latent distribution N(μ, σ²)
  - Returns mean (μ) and log-variance (log σ²) for 3D latent space
- **Decoder**: Reconstructs obstacle params from latent code
- **Reparameterization trick**: z = μ + ε·σ, where ε ~ N(0,1)

**B. Policy Network:**
- Similar MLP structure but takes latent code instead of raw params
- **Input**: state (3D) + goal (2D) + latent z (3D) = 8D total
- **Output**: action (2D)

**Two-Stage Training:**

**Stage 1: VAE Training**
```
Objective: Learn compressed obstacle representation
Loss = Reconstruction_Loss + KL_Divergence
     = MSE(decoded_c, original_c) + KL(q(z|c) || p(z))
     
Where:
- Reconstruction ensures z captures obstacle info
- KL regularization forces z ~ N(0,1) distribution
```

**Stage 2: Policy Training (VAE frozen)**
```
1. Encode obstacles: μ, log_var = VAE.encode(c)
2. Sample K latent codes: z_k = reparameterize(μ, log_var)
3. Get K action predictions: a_k = Policy(s, g, z_k)
4. Average predictions: a_final = mean(a_1, ..., a_K)
5. Minimize: Loss = MSE(a_final, expert_action)
```

**Key Innovation - Stochastic Encoding:**
- During training: Samples multiple z from the distribution
- Creates **implicit data augmentation** - same obstacle produces varied latent codes
- Forces policy to be robust to variations in latent space
- Acts as regularization against overfitting to exact obstacle configurations

**Mathematical Formulation:**
```
VAE: c → q(z|c) = N(μ(c), σ²(c))
Policy: π(s, g, z) → a
Training: E_z~q(z|c)[π(s, g, z)] ≈ expert_action
```

**Strengths:**
- **Superior generalization**: Latent space smoothness helps with unseen obstacles
- **Uncertainty modeling**: Stochastic sampling handles variability
- **Disentangled representations**: Latent code may capture abstract "danger" levels
- **Robustness**: Averaging over samples reduces prediction variance

**Weaknesses:**
- More complex architecture
- Two-stage training required
- Higher computational cost during inference (though can use single sample)
- Requires tuning latent dimension and sampling hyperparameters

---

### **Comparison Summary**

| Aspect | Behavior Cloning (PC-BC) | VAE Policy (RISE) |
|--------|-------------------------|-------------------|
| **Philosophy** | Direct imitation | Learn representations first |
| **Input** | Raw obstacle params (5D) | Learned latent code (3D) |
| **Training** | Single-stage supervised | Two-stage (VAE + Policy) |
| **Generalization** | Limited | Strong |
| **Complexity** | Low | Higher |
| **Best for** | Seen configurations | Novel scenarios |

The **RISE approach** is the core contribution: using stochastic latent encodings creates implicit robustness without requiring explicit adversarial training or domain randomization.
