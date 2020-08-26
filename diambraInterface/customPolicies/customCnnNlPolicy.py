from stable_baselines.common.policies import *

# Custom policy implementing a non linear policy from latent space to actions (for both policy and value networks)
class CustCnnNlPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param emb_layers: ([int]) The size of the Neural network for the policy and value net after embeddings
        (if None, default to [64, 64])
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=[64, 64], emb_layers = [64, 64],
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", n_add_info=148, **kwargs):
        super(CustCnnNlPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                            scale=False)

        self._kwargs_check(feature_extraction, kwargs)

        frames = self.processed_obs[:,:,:,0:self.processed_obs.shape[3]-1]
        additional_input = self.processed_obs[:,:,:,self.processed_obs.shape[3]-1]
        additional_input = tf.layers.flatten(additional_input)
        additional_input = additional_input[:,1:n_add_info+1]

        with tf.variable_scope("model", reuse=reuse):
            # Frames (CNN)
            extracted_features_frames = cnn_extractor(frames, **kwargs)

            # Additional (Additional Info)
            for i, layer_size in enumerate(layers):
                additional_input = act_fun(linear(additional_input, 'pi_fc' + str(i),
                                            n_hidden=layer_size, init_scale=np.sqrt(2)))

            pi_latent = vf_latent = tf.concat([extracted_features_frames, additional_input], 1)

            # MLP on embeddings, for Non Linear policy and value networks
            for i, layer_size in enumerate(emb_layers):
                pi_latent = act_fun(linear(pi_latent, 'pi_nl_fc' + str(i),
                                    n_hidden=layer_size, init_scale=np.sqrt(2)))
                vf_latent = act_fun(linear(vf_latent, 'vf_nl_fc' + str(i),
                                    n_hidden=layer_size, init_scale=np.sqrt(2)))

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

