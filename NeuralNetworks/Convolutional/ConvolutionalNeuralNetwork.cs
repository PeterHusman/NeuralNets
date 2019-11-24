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
            for (int i = 0; i < Layers.Length; i++)
            {
                vol = Layers[i].Compute(vol);
            }
            return vol;
        }

        public float GradientDescent(float learningRate, params (float[][][] input, float[][][] targetOutput)[] trainingData)
        {
            return GradientDescent(trainingData, learningRate);
        }

        public float GradientDescent((float[][][] input, float[][][] targetOutput)[] trainingData, float learningRate)
        {
            float[][][][] inputs = new float[trainingData.Length][][][];
            float[][][][] targetOutputs = new float[trainingData.Length][][][];
            for (int i = 0; i < trainingData.Length; i++)
            {
                inputs[i] = trainingData[i].input;
                targetOutputs[i] = trainingData[i].targetOutput;
            }
            return GradientDescent(inputs, targetOutputs, learningRate);
        }

        public float StochasticGradientDescent((float[][][] input, float[][][] targetOutput)[] trainingData, float learningRate)
        {
            float totalTotalError = 0f;
            for (int i = 0; i < trainingData.Length; i++)
            {
                totalTotalError += GradientDescent(learningRate, (trainingData[i].input, trainingData[i].targetOutput));
            }
            return totalTotalError;
        }

        public float StochasticGradientDescent(float[][][][] input, float[][][][] targetOutput, float learningRate)
        {
            float totalTotalError = 0f;
            for (int i = 0; i < input.Length; i++)
            {
                totalTotalError += GradientDescent(learningRate, (input[i], targetOutput[i]));
            }
            return totalTotalError;
        }

        public static float[] SoftArgMax(float[] ins)
        {
            float[] exps = ins.Select(a => (float)Math.Exp(a)).ToArray();
            float sum = exps.Sum();
            return exps.Select(a => a / sum).ToArray();
        }

        public static int IndexOfMax(float[] vals)
        {
            int ind = 0;
            for (int i = 0; i < vals.Length; i++)
            {
                if (vals[i] > vals[ind])
                {
                    ind = i;
                }
            }
            return ind;
        }

        private static float Kronecker(float a, float b)
        {
            return a == b ? 1f : 0f;
        }

        public static float[] SoftMaxDeltasFromOuts(float[] distr, int index)
        {
            float[] derivatives = new float[distr.Length];
            for (int i = 0; i < distr.Length; i++)
            {
                derivatives[i] = distr[index] * (Kronecker(i, index) - distr[i]);
            }
            return derivatives;
        }

        public float[] ComputeWithSoftMax(float[][][] input)
        {
            if (Layers.Last().OutputSideLength != 1)
            {
                throw new InvalidOperationException();
            }
            float[] output = Compute(input).Select(a => a[0][0]).ToArray();
            return SoftArgMax(output);
        }

        public int Categorize(float[][][] input)
        {
            return IndexOfMax(ComputeWithSoftMax(input));
        }

        public float GradientDescentWithSoftMax(float[][][][] input, int[] targetOut, float learningRate)
        {

            float totalError = 0f;

            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].ClearUpdates();
            }

            for (int inNumber = 0; inNumber < input.Length; inNumber++)
            {

                float[] outs = ComputeWithSoftMax(input[inNumber]);
                float[] outDeltas = SoftMaxDeltasFromOuts(outs, targetOut[inNumber]);
                float error = 1f - outs[targetOut[inNumber]];
                totalError += error;
                float[][][] errors = new float[Layers[Layers.Length - 1].Depth][][];
                for (int i = 0; i < errors.Length; i++)
                {
                    errors[i] = new[] { new[] { error * outDeltas[i] } };
                }

                errors = Layers[Layers.Length - 1].BackPropagation(errors);

                for (int i = Layers.Length - 2; i >= 0; i--)
                {
                    errors = Layers[i].BackPropagation(errors, Layers[i + 1]);
                }

            }


            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].ApplyUpdates(-learningRate);
            }

            return totalError;
        }

        public float GradientDescent(float[][][][] input, float[][][][] targetOut, float learningRate)
        {

            float totalError = 0f;

            for (int i = 0; i < Layers.Length; i++)
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

                errors = Layers[Layers.Length - 1].BackPropagation(errors);

                for (int i = Layers.Length - 2; i >= 0; i--)
                {
                    errors = Layers[i].BackPropagation(errors, Layers[i + 1]);
                }

            }


            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].ApplyUpdates(-learningRate);
            }

            return totalError;
        }

        public void Randomize(Random random)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Randomize(random);
            }
        }
    }
}
