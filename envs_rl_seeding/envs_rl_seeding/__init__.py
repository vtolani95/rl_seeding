from gym.envs.registration import register

register(
    id='SocialNetworkGraphEnv-v1',
    entry_point='envs_rl_seeding.envs.v0.social_network_env_v1:SocialNetworkGraphEnv'
)
