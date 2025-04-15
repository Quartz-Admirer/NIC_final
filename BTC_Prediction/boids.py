import numpy as np
import pandas as pd

class Boid:
    """
    Класс, описывающий одну "рыбу" (boid).
    Параметры:
    - position: np.array [x, y] – позиция
    - velocity: np.array [vx, vy] – скорость
    """
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

class BoidsSimulation:
    """
    Класс для запуска симуляции стаи Boids.
    Правила (стандартные):
    - separation (разделение): держаться на расстоянии
    - alignment (синхронизация): стремиться к средней скорости соседей
    - cohesion (стягивание): стремиться к центру масс соседей
    
    Теперь симуляция адаптируется к рыночному сигналу (market_signal):
      - При положительном тренде (trend > 0) увеличиваем cohesion и alignment,
      - При отрицательном тренде уменьшаем их,
      - При высокой волатильности увеличиваем separation.
    """
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

        # Инициализируем боидов случайно
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
        """
        Обновляет состояние всех боидов на один шаг симуляции.
        Если передан market_signal (словарь с ключами "trend" и "volatility"),
        то параметры симуляции адаптируются:
          - trend > 0: усиливаем cohesion и alignment,
          - trend < 0: ослабляем cohesion и alignment,
          - высокая volatility: увеличиваем separation.
        """
        # Адаптация параметров на основе рыночного сигнала
        if market_signal is not None:
            trend = market_signal.get("trend", 0)
            volatility = market_signal.get("volatility", 0)
            # Простейшая адаптация: значения подобраны экспериментально
            if trend > 0:
                self.cohesion_weight = 1.5
                self.alignment_weight = 1.2
            elif trend < 0:
                self.cohesion_weight = 0.8
                self.alignment_weight = 0.5
            else:
                # Нейтральное состояние
                self.cohesion_weight = 1.0
                self.alignment_weight = 1.0

            if volatility > 0.01:  # пороговое значение волатильности
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

            # Если нет соседей, движемся по текущей скорости
            if not neighbors:
                new_positions.append(boid.position + boid.velocity)
                new_velocities.append(boid.velocity)
                continue

            # Separation
            separation_force = np.zeros(2)
            for other_boid in neighbors:
                dist = self._distance(boid.position, other_boid.position)
                if dist > 0:
                    separation_force += (boid.position - other_boid.position) / dist
            separation_force *= self.separation_weight

            # Alignment
            avg_velocity = np.mean([other_boid.velocity for other_boid in neighbors], axis=0)
            alignment_force = (avg_velocity - boid.velocity) * self.alignment_weight

            # Cohesion
            avg_position = np.mean([other_boid.position for other_boid in neighbors], axis=0)
            cohesion_force = (avg_position - boid.position) * self.cohesion_weight

            # Итоговая скорость
            velocity = boid.velocity + separation_force + alignment_force + cohesion_force
            velocity = self._limit_speed(velocity)

            new_positions.append(boid.position + velocity)
            new_velocities.append(velocity)

        # Обновляем позиции и скорости
        for i, boid in enumerate(self.boids):
            boid.position = new_positions[i]
            boid.velocity = new_velocities[i]

            # Тор-анимация: если вышли за границу, появляются с противоположной стороны
            if boid.position[0] < 0:
                boid.position[0] = self.width
            elif boid.position[0] > self.width:
                boid.position[0] = 0
            if boid.position[1] < 0:
                boid.position[1] = self.height
            elif boid.position[1] > self.height:
                boid.position[1] = 0

    def get_features(self):
        """
        Считает метрики стаи:
        - Средняя позиция (mean_x, mean_y)
        - Средняя скорость (mean_vx, mean_vy)
        - Стандартное отклонение позиций (std_x, std_y)
        - Стандартное отклонение скоростей (std_vx, std_vy)
        """
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
    """
    Генерирует для каждого из num_days набор метрик боидов.
    Если передан market_signals (список словарей длины num_days),
    то на каждом шаге симуляции применяется соответствующий сигнал.
    
    Возвращает DataFrame со столбцами:
    [boids_mean_x, boids_mean_y, boids_mean_vx, boids_mean_vy,
     boids_std_x, boids_std_y, boids_std_vx, boids_std_vy]
    """
    sim = BoidsSimulation(num_boids=num_boids,
                          width=width,
                          height=height,
                          max_speed=max_speed,
                          perception_radius=perception_radius)
    
    records = []
    for day in range(num_days):
        # Если market_signals передан, используем сигнал для данного дня
        signal = market_signals[day] if market_signals is not None and len(market_signals) == num_days else None
        sim.update(market_signal=signal)
        feats = sim.get_features()
        records.append(feats)

    df_boids = pd.DataFrame(records)
    return df_boids

if __name__ == "__main__":
    # Пример использования:
    # Создадим список рыночных сигналов для 5 дней (для теста)
    market_signals = [
        {"trend": 0.5, "volatility": 0.005},  # небольшой тренд вверх, низкая волатильность
        {"trend": -0.3, "volatility": 0.015}, # тренд вниз, высокая волатильность
        {"trend": 0.2, "volatility": 0.008},
        {"trend": 0.0, "volatility": 0.012},
        {"trend": -0.1, "volatility": 0.009},
    ]
    test_boids = generate_boids_features(num_days=5, num_boids=10, market_signals=market_signals)
    print("Boids features:\n", test_boids)
