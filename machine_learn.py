import numpy as np
def MLNN(Object):
    def _init_(self,sizes):
        self.num_of_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(r,1) for r in sizes]
        self.weights = [np.random.randn(r,c) for r,c in zip(sizes[:-1],sizes[1:])]
    def sigmoid(value):
        return 1/(1+np.exp(-value))
    def feedforward(self,ivector):
        for bias,weight in zip(self.biases,self.weights):
            out = self.sigmoid(np.dot(weight,ivector)+bias)
        return out
    def SGD(self,training_data,epochs,btchsz,eta,test_date=None):
        if test_data:n_test = len(test_data)
        n = len(training_data)
        for sess in xrange(epochs):
            random.shuffle(training_data)
            minbtchs = [training_data[k:k+btchsz] for k in xrange(0,n,btchsz)]
            for minbtch in minbtchs:
                self.update_mbtch(minbtch,eta)
            if test_data:
                print("Epoch {0}:{1}/{2}").format(sess,evaluate(test_data),n_test)
            else:
                print("Epoch {0} complete").format(sess)
    def update_mbtch(self,minbtch,eta):
        nabla_b = [np.zeroes(bias.shape) for bias in self.biases]
        nabla_w = [np.zeroes(weight.shape) for weight in self.weights]
        for i,o in minbtch:
            delta_nabla_b,delta_nabla_w=self.backprop(i,o)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_W,delta_nabla_w)]
            self.weights = [w-(eta/len(minbtch))*nw for w,nw in zip(self.weights,nabla_w)]
            self.biases = [b-(eta/len(minbtch))*nb for b,nb in zip(self.biases,nabla_b)]
    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,x)+b
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)
            #backward pass
            delta = self.cost_derivative(activations[-1],y)*sigma_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta,activations[-2].transpose())
            for k in xrange(2,self.num_of_layers):
                z = zs[-k]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-k+1].transpose(), delta) * sp
                nabla_b[-k] = delta
                nabla_w[-k] = np.dot(delta, activations[-k-1].transpose())
            return (nabla_b,nabla_w)
    def cost_derivative(self,out,y):
        return out - y
    def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))
    def evaluate(self,test_data):
        test_results = [np.argmax(self.feedforward(x),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    