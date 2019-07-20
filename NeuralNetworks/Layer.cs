using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.NeuralNetworks
{
    public class Layer
    {
        public Neuron[] Neurons;
        public double[] Output;

        public Layer(ActivationFunction actFunc, int inputCount, int neuronCount)
        {
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(actFunc, inputCount);
            }
            Output = new double[Neurons.Length];
        }

        public void Randomize(Random rand)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Randomize(rand);
            }
        }

        public double[] Compute(double[] input)
        {
            for(int i = 0; i < Neurons.Length; i++)
            {
                Output[i] = Neurons[i].Compute(input);
            }

            return Output;
        }
    }
}
