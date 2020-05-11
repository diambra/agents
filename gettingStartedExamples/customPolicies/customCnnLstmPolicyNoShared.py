from stable_baselines.common.policies import *
from stable_baselines.common.policies import RecurrentActorCriticPolicy

class CustomCnnLstmPolicyNoShared(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn,
                 layers_policy=[64, 64], layers_value=[64,64],
                 layer_norm=True, scale_in=True, feature_extraction="cnn", **kwargs):

        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(CustomCnnLstmPolicyNoShared, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=scale_in)


        self._kwargs_check(feature_extraction, kwargs)

       # Features extraction (Shared)

        with tf.variable_scope("model", reuse=reuse):

            # Frames (CNN)
            extracted_features = cnn_extractor(self.processed_obs, **kwargs)

            # LSTM on extracted features (Shared)

            #print("Extracted feat frames")
            #print(extracted_features_frames)
            #print("Extracted feat additional info")
            #print(extracted_features_addinfo)
            #print("Extracted feat")
            #print(extracted_features)
            input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            #print("input_sequence")
            #print(input_sequence)
            #print("dones_ph")
            #print(self.dones_ph)
            masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
            #print("masks")
            #print(masks)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            #print("rnn_output")
            #print(rnn_output)
            #input("Pause")
            rnn_output = seq_to_batch(rnn_output)

            # Non shared part of networks
            # Policy
            latent_policy = rnn_output
            for i, layer_size in enumerate(layers_policy):
                latent_policy = act_fun(linear(latent_policy, 'pi_fc' + str(i),
                                        n_hidden=layer_size, init_scale=np.sqrt(2)))

            # Value
            latent_value = rnn_output
            for i, layer_size in enumerate(layers_value):
                latent_value = act_fun(linear(latent_value, 'vf_fc' + str(i),
                                        n_hidden=layer_size, init_scale=np.sqrt(2)))


            value_fn = linear(latent_value, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)

        self._value_fn = value_fn

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

