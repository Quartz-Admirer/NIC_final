import numpy as np
import pandas as pd

class Boid:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

class BoidsSimulation:

    def __init__(self,
                 num_boids=20,
                 width=640,
                 height=480,
                 max_speed=5.0,
                 separation_weight=1.5,
                 alignment_weight=1.0,
                 cohesion_weight=1.0,
                 perception_radius=50.0):
        self.num_boids = num_boids
        self.width = width
        self.height = height
        self.max_speed = max_speed

        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.perception_radius = perception_radius

        self.boids = []
        for _ in range(num_boids):
            position = np.random.rand(2) * np.array([width, height])
            velocity = (np.random.rand(2) - 0.5) * 10
            self.boids.append(Boid(position, velocity))

    def _limit_speed(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            velocity = (velocity / speed) * self.max_speed
        return velocity

    def _distance(self, a, b):
        return np.linalg.norm(a - b)

    def update(self, market_signal=None):
        if market_signal is not None:
            trend = market_signal.get("trend", 0)
            volatility = market_signal.get("volatility", 0)

            if trend > 0:
                self.cohesion_weight = 1.5
                self.alignment_weight = 1.2
            elif trend < 0:
                self.cohesion_weight = 0.8
                self.alignment_weight = 0.5
            else:

                self.cohesion_weight = 1.0
                self.alignment_weight = 1.0

            if volatility > 0.01:
                self.separation_weight = 2.0
            else:
                self.separation_weight = 1.0

        new_positions = []
        new_velocities = []

        for i, boid in enumerate(self.boids):
            neighbors = []
            for j, other_boid in enumerate(self.boids):
                if i == j:
                    continue
                dist = self._distance(boid.position, other_boid.position)
                if dist < self.perception_radius:
                    neighbors.append(other_boid)

            if not neighbors:
                new_positions.append(boid.position + boid.velocity)
                new_velocities.append(boid.velocity)
                continue

            separation_force = np.zeros(2)
            for other_boid in neighbors:
                dist = self._distance(boid.position, other_boid.position)
                if dist > 0:
                    separation_force += (boid.position - other_boid.position) / dist
            separation_force *= self.separation_weight

            avg_velocity = np.mean([other_boid.velocity for other_boid in neighbors], axis=0)
            alignment_force = (avg_velocity - boid.velocity) * self.alignment_weight

            avg_position = np.mean([other_boid.position for other_boid in neighbors], axis=0)
            cohesion_force = (avg_position - boid.position) * self.cohesion_weight

            velocity = boid.velocity + separation_force + alignment_force + cohesion_force
            velocity = self._limit_speed(velocity)

            new_positions.append(boid.position + velocity)
            new_velocities.append(velocity)

        for i, boid in enumerate(self.boids):
            boid.position = new_positions[i]
            boid.velocity = new_velocities[i]

            if boid.position[0] < 0:
                boid.position[0] = self.width
            elif boid.position[0] > self.width:
                boid.position[0] = 0
            if boid.position[1] < 0:
                boid.position[1] = self.height
            elif boid.position[1] > self.height:
                boid.position[1] = 0

    def get_features(self):

        positions = np.array([b.position for b in self.boids])
        velocities = np.array([b.velocity for b in self.boids])

        mean_pos = positions.mean(axis=0)
        mean_vel = velocities.mean(axis=0)
        std_pos = positions.std(axis=0)
        std_vel = velocities.std(axis=0)

        return {
            "boids_mean_x": mean_pos[0],
            "boids_mean_y": mean_pos[1],
            "boids_mean_vx": mean_vel[0],
            "boids_mean_vy": mean_vel[1],
            "boids_std_x": std_pos[0],
            "boids_std_y": std_pos[1],
            "boids_std_vx": std_vel[0],
            "boids_std_vy": std_vel[1],
        }

def generate_boids_features(num_days,
                            num_boids=20,
                            width=640,
                            height=480,
                            max_speed=5.0,
                            perception_radius=50.0,
                            market_signals=None):
 
    sim = BoidsSimulation(num_boids=num_boids,
                          width=width,
                          height=height,
                          max_speed=max_speed,
                          perception_radius=perception_radius)
    
    records = []
    for day in range(num_days):
        signal = market_signals[day] if market_signals is not None and len(market_signals) == num_days else None
        sim.update(market_signal=signal)
        feats = sim.get_features()
        records.append(feats)

    df_boids = pd.DataFrame(records)
    return df_boids

if __name__ == "__main__":

    market_signals = [
        {"trend": 0.5, "volatility": 0.005},
        {"trend": -0.3, "volatility": 0.015}, 
        {"trend": 0.2, "volatility": 0.008},
        {"trend": 0.0, "volatility": 0.012},
        {"trend": -0.1, "volatility": 0.009},
    ]
    test_boids = generate_boids_features(num_days=5, num_boids=10, market_signals=market_signals)
    print("Boids features:\n", test_boids)
