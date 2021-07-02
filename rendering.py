import numpy as np
import imageio
import gym
import mujoco_py as mjc
import pdb

class Renderer:

    def __init__(self, env):
        if type(env) is str:
            self.env = gym.make(env)
        else:
            self.env = env
        self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)

    def pad_observation(self, observation):
        state = np.concatenate([
            [np.random.uniform(0, 20)], #np.zeros(1),
            observation,
        ])
        return state

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None):

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        if not qvel:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        if type(dim) == int:
            dim = (dim, dim)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def composite(self, savepath, *args, **kwargs):
        sample_images = self.renders(*args, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        imageio.imsave(savepath, composite)
        return composite

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    assert state.size == qpos_dim + qvel_dim

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

