from gym.envs.registration import register

register(
    id='mazebase-v0',
    max_episode_steps=100,
    entry_point='mazebaseenv.envs:MazebaseGame',
)
