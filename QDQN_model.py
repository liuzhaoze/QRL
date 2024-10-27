import tensorflow as tf
import tensorflow_quantum as tfq
from collections import deque, Counter
import matplotlib.pyplot as plt
from args_config import args
import copy
import subprocess
import numpy as np
from env import SchedulingEnv
from cirq.contrib.svg import SVGCircuit
import cirq, sympy
from tensorflow.keras.utils import plot_model
tf.get_logger().setLevel('ERROR')

def get_gpu_info():
    try:
        return subprocess.check_output(["nvidia-smi"]).decode("utf-8")
    except Exception as e:
        return str(e)
    
def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def generate_circuit(qubits, n_layers):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_qubits})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)

class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """
    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )


        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)        

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])
    

class Rescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Rescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))    
    
def generate_model_Qlearning(n_qubits, n_layers, n_actions, target):
    """
    Generates a Keras model for a data re-uploading PQC Q-function approximator.
    """
    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    observables = [ ops[0]*ops[1],ops[1]*ops[2],ops[2]*ops[3],ops[3]*ops[4],ops[4]*ops[5],ops[5]*ops[6],ops[6]*ops[7],ops[7]*ops[8],ops[8]*ops[9],ops[9]*ops[10]]#observables = [ops[i] * ops[i + 1] for i in range(n_qubits - 1)] # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
    input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')([input_tensor])
    process = tf.keras.Sequential([Rescaling(len(observables))], name=target*"Target"+"Q-values")
    # 这里的Rescaling返回的对象是一个层而非张量
    Q_values = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)
    return model

optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    # Assign the model parameters to each optimizer
w_in, w_var, w_out = 1, 0, 2

@tf.function
def Q_learning_update(states, actions, rewards, next_states, model, model_target, gamma, n_actions, opt_in, opt_var, opt_out):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    rewards = tf.convert_to_tensor(rewards)
    next_states = tf.convert_to_tensor(next_states)

    # Compute their target q_values and the masks on sampled actions
    Q_target = model_target([next_states])
    target_q_values = rewards + (gamma * tf.reduce_max(Q_target, axis=1))
    masks = tf.one_hot(actions, n_actions)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        q_values = model([states])
        q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

    return loss

def generate_memory_underfitting(env, queue_len, args): 
    replay_memory = deque(maxlen = queue_len)
    job_c = 1
    step_count = 0
    print("*******************generate_Dataset*******************")
    for i in range(1, 6):
        job_c = 1
        step_count = 0
        env.reset(args, 4000, 10*i, 0.5)
        while True:
            step_count += 1 
            finish, job_attrs = env.workload(job_c)
            QDQN_state = env.getState( job_attrs, 4)

            if step_count != 1:
                replay_memory.append({'state': last_state,
                                                'action': last_action,
                                                'next_state': QDQN_state,
                                                'reward': float(last_reward)})
            action_QDQN = np.random.choice(args.n_actions)
            reward_QDQN = env.feedback(job_attrs, action_QDQN, 4)
            last_state = QDQN_state
            last_action = action_QDQN
            last_reward = reward_QDQN

            job_c += 1
            if job_c % 500 == 0:
                print("Arrive_Rate",10*i,"num",job_c)
            if finish:
                break
    return replay_memory

def generate_memory(env, TrainJobNum, TrainArriveRate, TrainJobType, args): 
    env.reset(args, TrainJobNum, TrainArriveRate, TrainJobType)
    replay_memory = deque(maxlen = 18000)
    job_c = 1
    step_count = 0
    print("*******************generate_Dataset*******************")
    while True:
        step_count += 1 
        finish, job_attrs = env.workload(job_c)
        QDQN_state = env.getState( job_attrs, 4)

        if step_count != 1:
            replay_memory.append({'state': last_state,
                                              'action': last_action,
                                              'next_state': QDQN_state,
                                              'reward': float(last_reward)})
        action_QDQN = np.random.choice(args.n_actions)
        reward_QDQN = env.feedback(job_attrs, action_QDQN, 4)
        last_state = QDQN_state
        last_action = action_QDQN
        last_reward = reward_QDQN

        job_c += 1
        if job_c % 1000 == 0:
            print(job_c)
        if finish:
            return replay_memory
        
def QDQN_train(env, model, model_target, replay_memory, batch_size, gamma, target_update, iteration, args):
    loss_list = []
    for episode in range( args.Epoch ):
        print("*******************TrainEpisode:",episode,"*******************")
        epoch = 1
        # env.reset(args)
        while True:            
            training_batch = np.random.choice(replay_memory, size=batch_size)
            loss = Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                np.asarray([x['action'] for x in training_batch]),
                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                np.asarray([x['next_state'] for x in training_batch]),
                                model, model_target, gamma, args.n_actions, 0.001, 0.001, 0.1)
            loss_list.append(loss)
            # Update target model
            if epoch % target_update == 0:
                model_target.set_weights(model.get_weights())
                        # print(training_batch)
                # if step_count % steps_to_print_weights == 0:
                #         weights = model.get_weights()
                #         for i, w in enumerate(weights):
                #                 print(f"Layer {i} weights: {w}")
                #epsilon = max(epsilon * decay_epsilon, epsilon_min)
            epoch += 1
            if epoch % 20 == 0:
                loss_mean = round(np.mean(loss_list[-20:]), 4)
                loss_var = round(np.var(loss_list[-20:]), 4)
                print("last[",epoch,"]iterations loss mean:",loss_mean,"var:",loss_var)
            if  epoch == iteration:
                break
    return model

def QDQN_test(model, env, job_num, lamda, job_type):
    episode_reward_history = []
    action_list= []
    total_reward = 0
    print("*******************testing*******************")
    job_c = 1
    env.reset(args, job_num, lamda, job_type)
    while True:   
               
        finish, job_attrs = env.workload(job_c)
        QDQN_state = env.getState( job_attrs, 2)
        q_vals = model([tf.convert_to_tensor([QDQN_state])])
        action_QDQN = int(tf.argmax(q_vals[0]).numpy())

        reward_QDQN = env.feedback(job_attrs, action_QDQN, 2)

        last_reward = reward_QDQN
        total_reward += last_reward
        action_list.append(action_QDQN)
        episode_reward_history.append(total_reward)
                
        job_c += 1
        if job_c % 500 == 0:
            print(job_c)
        if finish:
            print(action_list)
            total_success = env.get_totalSuccess(args.Baseline_num,1)
            avg_allRespTs = env.get_total_responseTs(args.Baseline_num,1)
            avg_util = env.get_avgUtilitizationRate(args.Baseline_num, 1)
            total_cost = env.get_totalCost(args.Baseline_num, 1)
            print ('responseT:',avg_allRespTs[1])
            print('success_rate:',total_success[1])
            print('utilizationRate:',avg_util[1])
            print('cost:',total_cost[1])
            frequency_counter = Counter(action_list)

            # 准备数据
            labels = list(range(10))
            counts = [frequency_counter.get(i, 0) for i in labels]

            # 绘图
            plt.figure(figsize=(10, 6))
            plt.bar(labels, counts, color='blue', alpha=0.7, label='Frequency')
            plt.xlabel('数字')
            plt.ylabel('频率')
            plt.title('action_list中各数字的频率')
            plt.xticks(labels)
            plt.yticks(range(max(counts) + 1))
            plt.legend()
            plt.show()
            return avg_allRespTs[1], total_success[1], avg_util[1], total_cost[1]

            