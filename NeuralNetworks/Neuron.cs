using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.NeuralNetworks
{
    public class Neuron
    {
        public double Bias;
        public double[] Weights;
        public double[] WeightUpdates;
        public double BiasUpdate;
        public double Output;
        public double Input;
        public double PartialDerivative;
        public ActivationFunction ActivationFunction;

        public Neuron(ActivationFunction actFunc, int inputCount)
        {
            Weights = new double[inputCount];
            WeightUpdates = new double[inputCount];
            ActivationFunction = actFunc;
        }

        public void Randomize(Random rand)
        {
            for(int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = rand.NextDouble() - 0.5;
            }
            Bias = rand.NextDouble() - 0.5;
        }

        public double Compute(double[] input)
        {
            Output = Bias;
            for(int i = 0; i < input.Length; i++)
            {
                Output += Weights[i] * input[i];
            }

            Input = Output;

            Output = ActivationFunction.Function(Output);

            return Output;
        }
    }
}
