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

        public ConvolutionalNeuralNetwork(params ICNNLayer[] layers)
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

        public float GradientDescent(float[][][][] input, float[][][][] targetOut, float learningRate)
        {
            
            float totalError = 0f;

            for(int i = 0; i < Layers.Length; i++)
            {
                Layers[i].ClearUpdates();
            }

            for (int inNumber = 0; inNumber < input.Length; inNumber++)
            {

                float[][][] outs = Compute(input[inNumber]);
                float[][][] errors = new float[targetOut[inNumber].Length][][];

                for (int i = 0; i < errors.Length; i++)
                {
                    errors[i] = new float[targetOut[inNumber][i].Length][];
                    for (int j = 0; j < errors[i].Length; j++)
                    {
                        errors[i][j] = new float[targetOut[inNumber][i][j].Length];
                        for (int k = 0; k < errors[i][j].Length; k++)
                        {
                            errors[i][j][k] = targetOut[inNumber][i][j][k] - outs[i][j][k];
                            totalError += Math.Abs(errors[i][j][k]);
                        }
                    }
                }

                for (int i = Layers.Length - 1; i >= 0; i--)
                {
                    errors = Layers[i].BackPropagation(errors);
                }

            }


            for(int i = 0; i < Layers.Length; i++)
            {
                Layers[i].ApplyUpdates(-learningRate);
            }

            return totalError;
        }

        public void Randomize(Random random)
        {
            for(int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Randomize(random);
            }
        }
    }
}
