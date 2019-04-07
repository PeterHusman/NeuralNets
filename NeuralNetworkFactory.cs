using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNets.NeuralNetworks;

namespace NeuralNets
{
    public class NeuralNetworkFactory
    {
        public FeedForwardNeuralNetwork CreateRandomizedFeedForwardNeuralNetwork(Random rand, int numOfInputs, params (int, Func<float, float>)[] layerInfo)
        {
            int numOfLayers = layerInfo.Length;
            int[] layerLengths = layerInfo.Select(x => x.Item1).ToArray();
            float[][][] weights = new float[numOfLayers][][];
            for (int i = 0; i < numOfLayers; i++)
            {
                weights[i] = new float[i == 0 ? numOfLayers : layerLengths[i - 1]][];
                for (int j = 0; j < weights[i].Length; j++)
                {
                    weights[i][j] = new float[layerLengths[i]];
                }
            }
            var net = new FeedForwardNeuralNetwork(new float[1][][], layerInfo.Select(x => x.Item2).ToArray());
            net.RandomizeWeights(rand ?? new Random());
            return net;
        }


    }
}
