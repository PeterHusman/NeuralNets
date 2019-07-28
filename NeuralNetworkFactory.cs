using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNets.NeuralNetworks;

namespace NeuralNets
{
    public static class NeuralNetworkFactory
    {
        public static void TrainGenetic((FeedForwardNetwork net, double fitness)[] population, Random random, double mutationRate, double preserved = 0.1, double crossedOver = 0.9)
        {
            Array.Sort(population, (a, b) => b.fitness.CompareTo(a.fitness));

            int start = (int)(population.Length * preserved);
            int end = (int)(population.Length * crossedOver);

            for(int i = start; i < end; i++)
            {
                Crossover(population[random.Next(start)].net, population[i].net, random);
                Mutate(population[i].net, random, mutationRate);
            }

            for(int i = end; i < population.Length; i++)
            {
                population[i].net.Randomize(random);
            }
        }

        private static void Mutate(FeedForwardNetwork network, Random random, double mutationRate)
        {
            foreach(Layer layer in network.Layers)
            {
                foreach(Neuron neuron in layer.Neurons)
                {
                    for(int i = 0; i < neuron.Weights.Length; i++)
                    {
                        neuron.Weights[i] += random.NextDouble() * 2 - 1;
                    }

                    neuron.Bias += random.NextDouble() * 2 - 1;
                }
            }
        }

        private static void Crossover(FeedForwardNetwork winner, FeedForwardNetwork child, Random random)
        {
            for(int i = 0; i < winner.Layers.Length; i++)
            {
                Layer winnerLayer = winner.Layers[i];
                Layer childLayer = child.Layers[i];
                int chiasmata = random.Next(winnerLayer.Neurons.Length);
                bool flip = random.Next(2) == 0;

                for(int j = (flip ? 0 : chiasmata); j < (flip ? chiasmata : winnerLayer.Neurons.Length); j++)
                {
                    Neuron winNeuron = winnerLayer.Neurons[j];
                    Neuron childNeuron = childLayer.Neurons[j];

                    winNeuron.Weights.CopyTo(childNeuron.Weights, 0);
                    childNeuron.Bias = winNeuron.Bias;
                }
            }
        }

        public static FeedForwardNeuralNetwork CreateRandomizedFeedForwardNeuralNetwork(Random rand, int numOfInputs, params (int, Func<float, float>)[] layerInfo)
        {
            int numOfLayers = layerInfo.Length;
            int[] layerLengths = layerInfo.Select(x => x.Item1).ToArray();
            float[][][] weights = new float[numOfLayers][][];
            for (int i = 0; i < numOfLayers; i++)
            {
                weights[i] = new float[(i == 0 ? numOfInputs : layerLengths[i - 1]) + 1][];
                for (int j = 0; j < weights[i].Length; j++)
                {
                    weights[i][j] = new float[layerLengths[i]];
                }
            }
            var net = new FeedForwardNeuralNetwork(weights, layerInfo.Select(x => x.Item2).ToArray());
            net.RandomizeWeights(rand ?? new Random());
            return net;
        }

        public static async Task RandomTrain(FeedForwardNeuralNetwork net, Random rand, float[][] inputs, float[][] outputs)
        {
            Matrix targetMatrix = new Matrix(outputs);
            while ((new Matrix(net.ComputeBatch(inputs)) - targetMatrix).Sum(a => a*a) != 0)
            {
                net.RandomizeWeights(rand);
            }
        }

        public static async Task GradientDescentTrain(FeedForwardNeuralNetwork net, float[][] inputs, float[][] desiredOutputs, float learningRate, float thresholdError, float decayBase = 1f)//, Func<float, float>[] derivatives)
        {
            while (net.GradientDescent(inputs, desiredOutputs, learningRate/*, derivatives*/) >= thresholdError)
            {
                learningRate *= decayBase;
            }
        }

        public static IEnumerable<float> GradientDescentTrainCoroutine(FeedForwardNeuralNetwork net, float[][] inputs, float[][] desiredOutputs, float learningRate, float thresholdError, float decayBase = 1f)//, Func<float, float>[] derivatives)
        {
            float error = 0f;
            do
            {
                error = net.GradientDescent(inputs, desiredOutputs, learningRate/*, derivatives*/);
                learningRate *= decayBase;
                yield return error;
            }
            while (error >= thresholdError);
        }
        public static IEnumerable<float> StochasticDescentTrainCoroutine(FeedForwardNeuralNetwork net, float[][] inputs, float[][] desiredOutputs, float learningRate, float thresholdError, float decayBase = 1f)//, Func<float, float>[] derivatives)
        {
            float error = 0f;
            do
            {
                error = 0f;
                for (int i = 0; i < inputs.Length; i++)
                {
                    error += net.GradientDescent(new[] { inputs[i] }, new[] { desiredOutputs[i] }, learningRate/*, derivatives*/);
                }
                error /= inputs.Length;
                learningRate *= decayBase;
                yield return error;
            }
            while (error >= thresholdError);
        }


    }
}
