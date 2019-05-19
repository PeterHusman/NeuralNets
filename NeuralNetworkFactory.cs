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

        public static async Task GradientDescentTrain(FeedForwardNeuralNetwork net, float[][] inputs, float[][] desiredOutputs, float learningRate)//, Func<float, float>[] derivatives)
        {
            while(net.GradientDescent(inputs, desiredOutputs, learningRate/*, derivatives*/) >= 7.251f)
            {

            }
        }


    }
}
