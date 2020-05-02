from stable_baselines.common.policies import *
from stable_baselines.common.policies import RecurrentActorCriticPolicy

class CustomMobLstmPolicy(RecurrentActorCriticPolicy):
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
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False,
                 layers=None, act_fun=tf.tanh, encoder_embeddings_n=512,
                 layer_norm=True, feature_extraction="mob", **kwargs):

        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(CustomMobLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=False)


        self._kwargs_check(feature_extraction, kwargs)

        assert ( (self.processed_obs.shape[1] == 224) and (self.processed_obs.shape[2] == 224) ), "Only 224 x 224 input images allowed"
        assert self.processed_obs.shape[3] == 4, "Only 4 channels allowed, 3 for the image, 1 for additional data"

        IMG_SHAPE = [224, 224, 3]

        self.encoderModel = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

        self.encoderModel.trainable = False

        frames = self.processed_obs[:,:,:,0:3]
        additional_input = self.processed_obs[:,:,:,3]
        additional_input = tf.layers.flatten(additional_input)
        additional_input = additional_input[:,1:149]

        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):

            # Frames (CNN)
            extracted_features_frames = self.encoderModel(frames)
            extracted_features_frames = conv_to_fc(extracted_features_frames)
            extracted_features_frames = tf.nn.relu(linear(extracted_features_frames, 'fc1',  n_hidden=encoder_embeddings_n, init_scale=np.sqrt(2)))

            # Additional (Additional Info)
            for i, layer_size in enumerate(layers):
                additional_input = act_fun(linear(additional_input, 'pi_fc' + str(i),
                                            n_hidden=layer_size, init_scale=np.sqrt(2)))

            extracted_features = tf.concat([extracted_features_frames, additional_input], 1)

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
            value_fn = linear(rnn_output, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

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

