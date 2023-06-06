import tensorflow as tf

def Flags(option_type):
    tf.compat.v1.app.flags.DEFINE_string("env_type", "gym", "environment type")
    tf.compat.v1.app.flags.DEFINE_string("env_name", "SuperMarioBros-v0",  "environment name")
    tf.compat.v1.app.flags.DEFINE_boolean("use_pixel_change", True, "whether to use pixel change")
    tf.compat.v1.app.flags.DEFINE_boolean("use_reward_prediction", True, "whether to use reward prediction")

    tf.compat.v1.app.flags.DEFINE_string("checkpoint_dir", "/tmp/a3c_checkpoints", "checkpoint directory")

    if option_type == 'training':
        tf.compat.v1.app.flags.DEFINE_integer("parallel_size", 8, "parallel thread size")
        tf.compat.v1.app.flags.DEFINE_integer("local_t_max", 20, "repeat step size")
        tf.compat.v1.app.flags.DEFINE_float("rmsp_alpha", 0.99, "decay parameter for rmsprop")
        tf.compat.v1.app.flags.DEFINE_float("rmsp_epsilon", 0.1, "epsilon parameter for rmsprop")

        tf.compat.v1.app.flags.DEFINE_string("log_file", "/tmp/a3c_log/a3c_log", "log file directory")
        tf.compat.v1.app.flags.DEFINE_float("initial_alpha_low", 1e-4, "log_uniform low limit for learning rate")
        tf.compat.v1.app.flags.DEFINE_float("initial_alpha_high", 5e-3, "log_uniform high limit for learning rate")
        tf.compat.v1.app.flags.DEFINE_float("initial_alpha_log_rate", 0.5, "log_uniform interpolate rate for learning rate")
        tf.compat.v1.app.flags.DEFINE_float("gamma", 0.99, "discount factor for rewards")
        tf.compat.v1.app.flags.DEFINE_float("gamma_pc", 0.9, "discount factor for pc")
        tf.compat.v1.app.flags.DEFINE_float("entropy_beta", 0.001, "entropy regularization constant")
        tf.compat.v1.app.flags.DEFINE_float("pixel_change_lambda", 0.05, "pixel change lambda")
        tf.compat.v1.app.flags.DEFINE_integer("experience_history_size", 4000, "experience replay buffer size")
        tf.compat.v1.app.flags.DEFINE_integer("max_time_step", 10 * 10**6, "max time steps")
        tf.compat.v1.app.flags.DEFINE_integer("save_interval_step", 100 * 1000, "saving interval steps")
        tf.compat.v1.app.flags.DEFINE_float("grad_norm_clip", 40.0, "gradient norm clipping")

    if option_type == 'display':
        tf.compat.v1.app.flags.DEFINE_string("frame_save_dir", "/tmp/a3c_frames", "frame save directory")
        tf.compat.v1.app.flags.DEFINE_boolean("recording", False, "record movie")
        tf.compat.v1.app.flags.DEFINE_boolean("frame_saving", False, "save frames")

    return tf.compat.v1.app.flags.FLAGS