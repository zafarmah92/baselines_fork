import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from gym import spaces


class Runner(AbstractEnvRunner):

    def __init__(self, env, model,nsteps , icm = None):
        super().__init__(env=env, model=model, icm=icm, nsteps=nsteps)
        assert isinstance(env.action_space, spaces.Discrete), 'This ACER implementation works only with discrete action spaces!'
        assert isinstance(env, VecFrameStack)

        self.nact = env.action_space.n
        nenv = self.nenv
        self.nbatch = nenv * nsteps

        self.batch_ob_shape = (nenv*(nsteps+1),) + env.observation_space.shape # 84 but it should be 80
        
        # print("Runner -> self batch_ob_shape ", self.batch_ob_shape)
        # print(" Runner -> nenv {} nsteps  {} env.observation_space.shape {}".format( nenv ,nsteps, env.observation_space.shape ))
        self.obs = env.reset()
        # observation type
        self.obs_dtype = env.observation_space.dtype

        self.ac_dtype = env.action_space.dtype
        self.nstack = self.env.nstack
        self.nc = self.batch_ob_shape[-1] // self.nstack


    def run(self):
        curiosity = False
        # curiosity = True
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        enc_next_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)


        # print("run function : encoded obsershape ",np.shape(enc_obs) )
        # enc_next_obs = np.split(self.env.stackedobs , self.env.env.nstack , axis=-1)
        # enc_next_obs = enc_obs
        # print("run function : encoded obsershape ",np.shape(enc_next_obs) )
        
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards , mb_next_states = [] , [], [], [], [], []
        for _ in range(self.nsteps):

            actions, mus, states = self.model._step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(np.copy(self.obs)) # s_t
            
            enc_obs.append(self.obs[..., -self.nc:]) # s_t
            
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)

            if curiosity == True :
                icm_states = self.obs

            obs, rewards, dones, _ = self.env.step(actions)
            # states information for statefull models like LSTM
            # print("Acer -> default Reward shape {}  and value {} ".format(
                    # np.shape(rewards) , rewards))
                
            if curiosity == True :
                icm_next_states = obs

                rewards = self.icm.calculate_intrinsic_reward(icm_states,icm_next_states,actions)
                # print("Acer -> curiosity intrensiv Reward shape {}  and value {} ".format(np.shape(rewards) , rewards))
                


            mb_next_states.append(np.copy(obs)) # s_t+1
            
            self.states = states
            self.dones = dones
            self.obs = obs
            self.reward=rewards 
            self.actions=actions
            mb_rewards.append(rewards)
            enc_next_obs.append(obs[..., -self.nc:]) # s_t+1
            # enc_next_obs.append()
        
        mb_obs.append(np.copy(self.obs))  # s_t+1
        mb_next_states.append(np.copy(self.obs)) #s_t+1
        
        # print("acer runner  -> encoded observation : ",enc_obs)
        mb_dones.append(self.dones) # s_t+1

        icm_actions = mb_actions 
        icm_rewards = mb_rewards


        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        # print("acer runner  -> encoded observation : ", enc_obs.shape)

        enc_next_obs = np.asarray(enc_next_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_next_states = np.array(mb_next_states, dtype=self.obs_dtype).swapaxes(1,0)

        icm_actions.append(actions)
        icm_rewards.append(rewards)
        icm_actions = np.asarray(icm_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        icm_rewards = np.asarray(icm_rewards, dtype=np.float32).swapaxes(1, 0)
         # print("changed the mb_next_states dimension" )
        # print(" encoded obs {} enc next obs {} next states obs {} encoded next states {}".format(
        #     enc_obs.shape , enc_next_obs.shape , mb_next_states.shape , "None"))

        # print("testing so far ..... ")

        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones # Used for statefull modefacels like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards

        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        # print(" Runner -> shape of the following rewards {}  dones {}".format(mb_rewards.shape , mb_dones.shape))
        # print("sent parameters enc obs {} enc next obs {} mb_obs {} actions {} mb_rewards {} ".format( 
        #     enc_obs.shape , enc_next_obs.shape ,mb_obs.shape, mb_actions.shape, mb_rewards.shape))
        
        return enc_obs, enc_next_obs ,mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks, mb_next_states, icm_actions , icm_rewards

