import tensorflow as tf
from tensorflow.python.training import *

class A3C_Mean_Squared_Error(object):

    def __init__(self,
               learning_rate,
               decay=0.9,
               momentum=0.0,
               epsilon=1e-10,
               clip_norm=40.0,
               device="/cpu:0",
               name="A3C_Mean_Squared_Error"):

        self.name = name
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.device = device
        self.learning_rate_tensor = None
        self.decay_tensor = None
        self.momentum_tensor = None
        self.epsilon_tensor = None

        self.slots = {}

    def mk_slots(self, variable_list):
        for v in variable_list:
            value = tf.compat.v1.constant(1.0, dtype=v.dtype, shape=v.get_shape())
            self.which_slot(v, value, "rms", self.name)
            self.slot_zeros(v, "momentum", self.name)

    def slot_prepare(self):
        self.learning_rate_tensor = tf.compat.v1.convert_to_tensor(self.learning_rate, name="learning_rate")
        self.decay_tensor = tf.compat.v1.convert_to_tensor(self.decay, name="decay")
        self.momentum_tensor = tf.compat.v1.convert_to_tensor(self.momentum, name="momentum")
        self.epsilon_tensor = tf.compat.v1.convert_to_tensor(self.epsilon, name="epsilon")

    def slots_dict(self, slotname):
        named_slots = self.slots.get(slotname, None)
        if named_slots is None:
          named_slots = {}
          self.slots[slotname] = named_slots
        return named_slots

    def which_slot(self, variable, value, slotname, opname):
        named_slots = self.slots_dict(slotname)
        if variable not in named_slots:
          named_slots[variable] = slot_creator.create_slot(variable, value, opname)
        return named_slots[variable]

    def slot_get(self, variable, name):
        named_slots = self.slots.get(name, None)
        if not named_slots:
          return None
        return named_slots.get(variable, None)

    def slot_zeros(self, variable, slotname, opname):
        named_slots = self.slots_dict(slotname)
        if variable not in named_slots:
          named_slots[variable] = slot_creator.create_zeros_slot(variable, opname)
        return named_slots[variable]

    def dense(self, grad, variable):
        rms = self.slot_get(variable, "rms")
        mom = self.slot_get(variable, "momentum")
        return training_ops.apply_rms_prop(
        variable, rms, mom,
        self.learning_rate_tensor,
        self.decay_tensor,
        self.momentum_tensor,
        self.epsilon_tensor,
        grad,
        use_locking=False).op

    def minimize_local(self, loss, global_variables, local_variables):
        with tf.device(self.device):
          variable_refs = [v.read_value() for v in local_variables]
          local_gradients = tf.compat.v1.gradients(
            loss, variable_refs,
            gate_gradients=False,
            aggregation_method=None,
            colocate_gradients_with_ops=False)
          return self.apply_gradients(global_variables, local_gradients)

    def apply_gradients(self, global_variables, local_grad_list, name=None):
        update_ops = []

        with tf.control_dependencies(None):
            self.mk_slots(global_variables)

        # global gradient norm clipping
        local_grad_list, _ =  tf.clip_by_global_norm(local_grad_list, self.clip_norm)

        with tf.compat.v1.variable_scope(name, self.name,[]) as name:
            self.slot_prepare()
            for variable, grad in zip(global_variables, local_grad_list):
                with tf.compat.v1.variable_scope("update_" + variable.op.name), tf.device(variable.device):
                    update_ops.append(self.dense(grad, variable))
            return tf.group(*update_ops, name=name)



#Error = TypeError: Variable is unhashable. Instead, use variable.ref() as the key. (Variable: <tf.Variable 'Variable:0' shape=(2,)
