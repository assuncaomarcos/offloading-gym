import keras 

class ModelConfig:
  def __init__(self):
    self.h = 8
    self.d_k = 512
    self.d_v = 512
    self.d_ff = 512
    self.d_model = 512
    self.n = 3
    self.dropout_rate = 0.001
    self.steps_per_epoch = 1000
    self.gamma = 0.99 #discount_factor
    self.clip_ratio = 0.2
    self.max_gradient_norm = 0.7
    self.policy_learning_rate = 5e-4
    self.value_function_learning_rate = 5e-4
    self.train_policy_iterations = 100
    self.train_value_iterations = 100
    self.lam = 0.97
    self.target_kl = 0.3
    self.hidden_sizes = (512, 512)
    self.action_dim = 20
    self.num_actions = 1
    self.obs_dim = (20, 17)
    self.entropy_coef = 0.2
    self.meta_batch_size = 10
    self.vf_coef = 0.5
    self.train_save_interval = 100
    self.model_save_path = './checkpoints'
    self.inner_lr = 0.4
    self.outer_lr = 0.2
    self.policy_optimizer = keras.optimizers.legacy.Adam(self.policy_learning_rate, epsilon=1e-5)
    
    #schedules.PolynomialDecay parameters 
    self.initial_learning_rate = 0.1  
    self.total_steps = 1000000  
    self.end_learning_rate = 0.0  
    self.power = 1.0 


   