using System;
using System.Linq;

namespace NeuralNets.NeuralNetworks
{
    public class FeedForwardNeuralNetwork
    {
        public (Matrix, Func<float, float>)[] Layers;
        
        public FeedForwardNeuralNetwork(float[][][] weights, Func<float, float>[] activationFunctions)
        {
            Layers = new (Matrix, Func<float, float>)[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                Layers[i] = (new Matrix(weights[i]), activationFunctions[i]);
            }
        }

        public void RandomizeWeights(Random rand)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                var layer = Layers[i].Item1;
                for (int j = 0; j < layer.Columns; j++)
                {
                    for (int k = 0; k < layer.GetColumn(j).Length; k++)
                    {
                        layer[k, j] = (float)rand.NextDouble() - 0.5f;
                    }
                }
            }
        }

        public float[] Compute(float[] inputs)
        {
            float[] outputs = inputs;
            for (int i = 0; i < Layers.Length; i++)
            {
                outputs = ComputeLayer(outputs, Layers[i].Item1, Layers[i].Item2);
            }

            return outputs;
        }

        static float[] ComputeLayer(float[] inputs, Matrix layerWeights, Func<float, float> activationFunction)
        {
            float[] insWithBias = new float[inputs.Length + 1];
            insWithBias[0] = 1;
            inputs.CopyTo(insWithBias, 1);
            var outputs = (layerWeights * new Matrix(new []{insWithBias})).GetColumn(0).Select(activationFunction);
            return (outputs as float[]) ?? outputs.ToArray();
        }

        public float[][] ComputeBatch(float[][] inputs)
        {
            float[][] outputs = inputs;
            for (int i = 0; i < Layers.Length; i++)
            {
                outputs = ComputeLayerBatch(outputs, Layers[i].Item1, Layers[i].Item2);
            }

            return outputs;
        }

        static float[][] ComputeLayerBatch(float[][] inputs, Matrix layerWeights, Func<float, float> activationFunction)
        {
            float[][] insWithBias = new float[inputs.Length][];
            for (int i = 0; i < insWithBias.Length; i++)
            {
                insWithBias[i] = new float[inputs[i].Length + 1];
                insWithBias[i][0] = 1;
                inputs[i].CopyTo(insWithBias[i], 1);
            }
            var outputs = (layerWeights * new Matrix(insWithBias)).Values.Select(x => x.Select(activationFunction));
            if (outputs is float[][] arr)
            {
                return arr;
            }

            var outs1 = outputs.ToArray();
            var outs2 = new float[outs1.Length][];
            for (var index = 0; index < outs1.Length; index++)
            {
                outs2[index] = outs1[index].ToArray();
            }

            return outs2;
        }
    }
}
