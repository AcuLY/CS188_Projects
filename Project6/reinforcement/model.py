import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        self.parameters = [ # 两层隐藏层
            nn.Parameter(state_dim, 300),
            nn.Parameter(1, 300),
            nn.Parameter(300, 100),
            nn.Parameter(1, 100),
            nn.Parameter(100, action_dim),
            nn.Parameter(1, action_dim)
        ]
        self.learning_rate = 0.5
        self.numTrainingGames = 3000
        self.batch_size = 100

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        if not Q_target:    # 处理无 Q_target 的情况
            return 0
        
        prediction = self.run(states)
        return nn.SquareLoss(prediction, Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        hidden1 = nn.Linear(states, self.parameters[0])
        hidden1 = nn.AddBias(hidden1, self.parameters[1])
        hidden1 = nn.ReLU(hidden1)
        
        hidden2 = nn.Linear(hidden1, self.parameters[2])
        hidden2 = nn.AddBias(hidden2, self.parameters[3])
        hidden2 = nn.ReLU(hidden2)
        
        output = nn.Linear(hidden2, self.parameters[4])
        output = nn.AddBias(output, self.parameters[5])
        
        return output

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        loss = self.get_loss(states, Q_target)
        gradients = nn.gradients(loss, self.parameters)
        
        for index, gradient in enumerate(gradients):
            self.parameters[index].update(gradient, -self.learning_rate)
