import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w) # 将 x 与当前模型的权重点积即为结果

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        dot_product = self.run(x)
        product = nn.as_scalar(dot_product)
        # 非负返回 1, 负数返回 0
        if product >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            all_correct = True  # 检查是否全部数据都估计正确
            # 遍历所有数据, 如果估计错误就朝正确的方向更新权重值
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                correct_label = nn.as_scalar(y)
                if not prediction == correct_label:
                    all_correct = False
                    self.w.update(x, correct_label) # 正确值即为向量的学习率(如果正确值是 1 就把数据向量加到权重向量上面, 否则减)
            # 全部都估计正确则结束
            if all_correct:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        # 设置一层隐藏层, 100 个神经元
        self.w1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 1)
        # 输出的矩阵为 1 * 1
        self.b1 = nn.Parameter(1, 100)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # 隐藏层线性变换与非线性处理
        hidden1 = nn.Linear(x, self.w1)
        hidden1 = nn.AddBias(hidden1, self.b1)
        hidden1 = nn.ReLU(hidden1)
        # 输出层线性变换
        output = nn.Linear(hidden1, self.w2)
        output = nn.AddBias(output, self.b2)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        # 设置训练批大小、学习率以及判断学习结束的可接受损失率
        batch_size = 100
        learning_rate = 0.01
        accetped_loss = 0.005
        # 无限循环, 直到损失率达到要求
        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)
            scalarized_loss = nn.as_scalar(loss)    # 将损失标量化用于判断

            if scalarized_loss < accetped_loss: # 判断是否结束学习
                return

            # 朝损失率最小化的方向更新所有权重参数
            gw1, gb1, gw2, gb2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
            self.w1.update(gw1, -learning_rate)
            self.b1.update(gb1, -learning_rate)
            self.w2.update(gw2, -learning_rate)
            self.b2.update(gb2, -learning_rate)
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.w1 = nn.Parameter(784, 500)    # 每个数据点的大小为 784, 使用 2 个隐藏层, 分别 500, 200 个神经元
        self.w2 = nn.Parameter(500, 200)
        self.w3 = nn.Parameter(200, 10) # 输出的长度为 10, 代表 10 个数字的可能性
        
        self.b1 = nn.Parameter(1, 500)
        self.b2 = nn.Parameter(1, 200)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        hidden1 = nn.Linear(x, self.w1)
        hidden1 = nn.AddBias(hidden1, self.b1)
        hidden1 = nn.ReLU(hidden1)
        
        hidden2 = nn.Linear(hidden1, self.w2)
        hidden2 = nn.AddBias(hidden2, self.b2)
        hidden2 = nn.ReLU(hidden2)
        
        output = nn.Linear(hidden2, self.w3)
        output = nn.AddBias(output, self.b3)
        
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        prediction = self.run(x)
        return nn.SoftmaxLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 120
        learning_rate = 0.1
        accetped_accuracy = 0.98

        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)

            gw1, gb1, gw2, gb2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
            self.w1.update(gw1, -learning_rate)
            self.b1.update(gb1, -learning_rate)
            self.w2.update(gw2, -learning_rate)
            self.b2.update(gb2, -learning_rate)
            
            if dataset.get_validation_accuracy() > accetped_accuracy:   # 使用 dataset 提供的方法判断准确率
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.hidden_num = 100         # 每层的神经元数量
        self.max_word_length = 30     # 可接受的最大单词长度, 即最大隐藏层数
        
        # 每层对当前字母的权重, 以及当前字母是最后一个字母时要将输出转为长度为 5 的输出值
        self.wxs = [nn.Parameter(self.num_chars, self.hidden_num)] * self.max_word_length
        self.wxs_end = [nn.Parameter(self.num_chars, len(self.languages))] * self.max_word_length
        # 对上一层隐藏层结果的权重, 同理
        self.whs = [nn.Parameter(self.hidden_num, self.hidden_num)] * self.max_word_length
        self.whs_end = [nn.Parameter(self.hidden_num, len(self.languages))] * self.max_word_length

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a   at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        outcome = None  # 初始化结果为 None
        l = len(xs)
        for i, x in enumerate(xs):  # 逐字母遍历输入列表
            if i == l - 1:  # 如果当前是最后一个字母, 单独处理
                z = nn.Linear(x, self.wxs_end[i]) # 对当前字母变换
                
                if not outcome: # 单词长度为 1 的情况
                    outcome = z
                else:
                    outcome = nn.Linear(outcome, self.whs_end[i])
                    outcome = nn.Add(outcome, z)
                    
                return outcome
            
            # 不是最后一个字母则作为隐藏层处理
            z = nn.Linear(x, self.wxs[i])

            if not outcome: # 处理第一个字母的情况
                outcome = nn.Linear(x, self.wxs[i])
            else:
                outcome = nn.Linear(outcome, self.whs[i])
                outcome = nn.Add(outcome, z)
            
            outcome = nn.ReLU(outcome)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        prediction = self.run(xs)
        return nn.SoftmaxLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 120
        learning_rate = 0.01
        accetped_accuracy = 0.85

        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)

            # 遍历所有权重参数, 计算相应的梯度
            gs = nn.gradients(loss, self.wxs + self.wxs_end + self.whs + self.whs_end)
            for i in range(self.max_word_length):
                self.wxs[i].update(gs[i], -learning_rate)
                self.wxs_end[i].update(gs[i + self.max_word_length], -learning_rate)
                
                if i > 0:   # 第一个字母没有隐藏层的权重
                    self.whs[i].update(gs[i + self.max_word_length * 2], -learning_rate)
                    self.whs_end[i].update(gs[i + self.max_word_length * 3], -learning_rate)
            
            if dataset.get_validation_accuracy() > accetped_accuracy:
                return
