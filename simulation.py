import gym
import numpy as np

class DroneEnv(gym.Env):
    """Custom Environment for controlling a drone"""
    
    def __init__(self):
        super(DroneEnv, self).__init__()
        # Визначаємо простір дій: вперед, назад, ліворуч, праворуч
        self.action_space = gym.spaces.Discrete(4)  # 4 действия
        # Пространство состояний (например, положение дрона и препятствий)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(5,), dtype=np.float32)
        
        # Початкова позиція дрону
        self.state = np.array([0, 0, 0, 0, 0])  # x, y, z, скорость, расстояние до препятствия
        self.goal = np.array([10, 10, 5])  # Цель - конечная точка
    
    def reset(self):
        """Скидання стану дрону на початку нового епізоду"""
        self.state = np.array([0, 0, 0, 0, 0])
        return self.state

    def step(self, action):
        """Застосування дії до дрону та обчислення нового стану"""
        x, y, z, speed, obstacle_distance = self.state
        
        # Оновлюємо положення дрона в залежності від дії
        if action == 0:  # вперед
            x += 1
        elif action == 1:  # назад
            x -= 1
        elif action == 2:  # влево
            y -= 1
        elif action == 3:  # вправо
            y += 1
        
        # Вважаємо відстань до мети
        distance_to_goal = np.linalg.norm(np.array([x, y, z]) - self.goal)
        
        # Перевіряємо зіткнення з перешкодою
        if obstacle_distance < 1.0:
            reward = -100  # штраф за столкновение
            done = True
        else:
            reward = -distance_to_goal  # вознаграждение основано на близости к цели
            done = distance_to_goal < 1.0  # если дрон достиг цели, эпизод заканчивается
        
        # Оновлюємо стан
        self.state = np.array([x, y, z, speed, obstacle_distance])
        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Відображення середовища для моніторингу
        print(f"Drone position: {self.state}, Goal: {self.goal}")
