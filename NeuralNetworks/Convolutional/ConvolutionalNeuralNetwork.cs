using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.NeuralNetworks.Convolutional
{
    public class ConvolutionalNeuralNetwork
    {
        public ICNNLayer[] Layers { get; private set; }

        public ConvolutionalNeuralNetwork(ICNNLayer[] layers)
        {
            Layers = layers;
        }

        public float[][][] Compute(float[][][] input)
        {
            float[][][] vol = input;
            for(int i = 0; i < Layers.Length; i++)
            {
                vol = Layers[i].Compute(vol);
            }
            return vol;
        }

        public void GradientDescent(float[][][] input, float[][][] targetOut, float learningRate)
        {
            float[][][] outs = Compute(input);
            float[][][] errors = new float[targetOut.Length][][];
            for(int i = 0; i < errors.Length; i++)
            {
                errors[i] = new float[targetOut[i].Length][];
                for(int j = 0; j < errors[i].Length; j++)
                {
                    errors[i][j] = new float[targetOut[i][j].Length];
                    for(int k = 0; k < errors[i][j].Length; k++)
                    {
                        errors[i][j][k] = targetOut[i][j][k] - outs[i][j][k];
                    }
                }
            }

            for(int i = Layers.Length - 1; i >= 0; i--)
            {
                errors = Layers[i].BackPropagation(errors, learningRate);
            }
        }
    }
}
