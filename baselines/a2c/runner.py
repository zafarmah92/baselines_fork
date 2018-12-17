import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
import itertools
import tensorflow as tf

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, icm ,nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, icm=icm  ,nsteps=nsteps)
        
        self.gamma = gamma
        self.icm = icm

        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        # print(" Runner init -> batch action shape ", self.batch_action_shape)
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        # curiosity = True
        curiosity = False

        mb_obs, mb_rewards, mb_extrinsic_reward, mb_actions, mb_values, mb_dones, mb_next_states = [],[],[],[],[],[],[]
        mb_states = self.states
        # epinfos=[]
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)




            # print("run step i {} , actions {} , values {} , states {} , obs {} "
            #      .format(n, np.shape(actions), np.shape(values),np.shape(states) , np.shape(self.obs)))

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            if curiosity == True :
                # all row
                icm_states = self.obs[:] 
                

            # Take actions in env and look the results
            # try :
                # print("Forworded to step function states : {} , actions {} "
                    # .format( len(icm_states) ,len(actions)))
            # except:
                # print("forward model has states of nontype ")
                # pass



            obs, rewards, dones, _ = self.env.step(actions)
            mb_extrinsic_reward.append(rewards)
            # for info in info :
            #     maybeepinfo = info.get('episode')
            #     if maybeepinfo: epinfos.append(maybeepinfo)


            # This step function is called in the monitor file 

            # self.env.render()

            if curiosity == True :
                icm_next_states = obs[:]
                # print("step rewards ",rewards)
                # rewards = 0.0
                # rewards = []
                # state, next_state, action
                # for i_state, i_next_state , i_action in zip( icm_states, icm_next_states, actions) :
                #     print("passed evn received rewards {} , state s_t {} , s_t+1 {} "
                #         .format( np.shape(rewards) ,np.shape(i_state) , np.shape(i_next_state)))
                #     print(np.shape(tf.reshape(i_state , [1,np.shape(i_state)])))
                #     rewards.append(self.icm.calculate_intrinsic_reward(i_state,i_next_state,i_action))
                    
                # print("Forwarded states ")
                # print(" s_t {} s_t+1 {} actions {}".format(len(icm_states) , len(icm_next_states) , len(actions)))
                rewards = self.icm.calculate_intrinsic_reward(icm_states,icm_next_states,actions)
                # print("received Reward ",len(rewards))
                # print("passed evn received rewards {} , state s_t {} , s_t+1 {} "
                #     .format(rewards ,np.shape(icm_states) , np.shape(icm_next_states)))
                
                # rewards = [rewards,rewards ,rewards]

                # print("Rewards received from icm Model : ", rewards)
                # loss = self.icm.train()

            mb_next_states.append(np.copy(self.obs))

            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        # print(" Batch op shape a2c : ",self.batch_ob_shape)
        mb_next_states = np.asarray(mb_next_states , dtype=self.ob_dtype).swapaxes(1,0).reshape(self.batch_ob_shape)
        # print("Rewards THat give errors :: ",mb_rewards)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        # print("Rewards after conversion of dymension ", mb_rewards)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        # print(" mb masks shape {} , mb masks {} ".format(np.shape(mb_masks), mb_masks))
        # print(" mb dones shape {} , mb dones {}".format(np.shape(mb_dones) , mb_dones ))

        if curiosity == True:
            if self.gamma > 0.0:
                # Discount/bootstrap off value fn
                last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
                for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                    rewards = rewards.tolist()
                    dones = dones.tolist()
                    # if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                    # else:
                    # rewards = discount_with_dones(rewards, dones, self.gamma)

                    mb_rewards[n] = rewards
                # print("Present Rewards ::: ", mb_rewards)
        else :

            if self.gamma > 0.0:
                # Discount/bootstrap off value fn
                last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
                for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                    rewards = rewards.tolist()
                    dones = dones.tolist()
                    if dones[-1] == 0:
                        rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                    else:
                        rewards = discount_with_dones(rewards, dones, self.gamma)

                    mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        # print(" Mini batch shapes  obs {} , states  {} , rewards {} ".format(np.shape(mb_obs) , type(mb_states) , np.shape(mb_rewards)))
        
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_next_states, mb_extrinsic_reward 
