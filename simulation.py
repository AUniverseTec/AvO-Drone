import gym
import numpy as np

class DroneEnv(gym.Env):
    """Custom Environment for controlling a drone"""
    
    def __init__(self):
        super(DroneEnv, self).__init__()
        # Определяем пространство действий: вперед, назад, влево, вправо
        self.action_space = gym.spaces.Discrete(4)  # 4 действия
        # Пространство состояний (например, положение дрона и препятствий)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(5,), dtype=np.float32)
        
        # Начальная позиция дрона
        self.state = np.array([0, 0, 0, 0, 0])  # x, y, z, скорость, расстояние до препятствия
        self.goal = np.array([10, 10, 5])  # Цель - конечная точка
    
    def reset(self):
        """Сброс состояния дрона при начале нового эпизода"""
        self.state = np.array([0, 0, 0, 0, 0])
        return self.state

    def step(self, action):
        """Применение действия к дрону и вычисление нового состояния"""
        x, y, z, speed, obstacle_distance = self.state
        
        # Обновляем положение дрона в зависимости от действия
        if action == 0:  # вперед
            x += 1
        elif action == 1:  # назад
            x -= 1
        elif action == 2:  # влево
            y -= 1
        elif action == 3:  # вправо
            y += 1
        
        # Считаем расстояние до цели
        distance_to_goal = np.linalg.norm(np.array([x, y, z]) - self.goal)
        
        # Проверяем столкновение с препятствием
        if obstacle_distance < 1.0:
            reward = -100  # штраф за столкновение
            done = True
        else:
            reward = -distance_to_goal  # вознаграждение основано на близости к цели
            done = distance_to_goal < 1.0  # если дрон достиг цели, эпизод заканчивается
        
        # Обновляем состояние
        self.state = np.array([x, y, z, speed, obstacle_distance])
        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Отображение среды для мониторинга
        print(f"Drone position: {self.state}, Goal: {self.goal}")
