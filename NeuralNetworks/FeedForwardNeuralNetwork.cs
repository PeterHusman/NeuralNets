using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNets.NeuralNetworks
{
    public class FeedForwardNeuralNetwork
    {
        //Columns = weights
        //Rows = neurons
        public (Matrix, Func<float, float>)[] Layers;

        private Matrix[] Inputs;
        private Matrix[] Outputs;
        private Matrix[] WeightUpdates;
        private Matrix[] PartialDerivatives;

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
            var outputs = (layerWeights * new Matrix(new[] { insWithBias })).GetColumn(0).Select(activationFunction);
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

        public float GradientDescent(float[][] inputs, float[][] desiredOutputs, float learningRate)
        {
            Func<float, float>[] layerDerivatives = new Func<float, float>[Layers.Length];
            for(int i = 0; i < layerDerivatives.Length; i++)
            {
                layerDerivatives[i] = ActivationFunctions.DefaultFunctionData[Layers[i].Item2];
            }
            return GradientDescent(inputs, desiredOutputs, learningRate, layerDerivatives);
        }

        public float GradientDescent(float[][] inputs, float[][] desiredOutputs, float learningRate, Func<float, float>[] layerDerivatives)
        {
            Inputs = new Matrix[Layers.Length];
            Outputs = new Matrix[Layers.Length];
            WeightUpdates = new Matrix[Layers.Length];
            PartialDerivatives = new Matrix[Layers.Length];
            Matrix output = ComputeBatchWithRecords(inputs);
            var errors = desiredOutputs - output;
            float outputError = errors.Sum(a => Math.Abs(a)) / (desiredOutputs.Length * desiredOutputs[0].Length);

            for (int j = 0; j < errors.Columns; j++)
            {

                //Bug here! The sizes are wonky.
                PartialDerivatives[Layers.Length - 1] = Layers.Last().Item1.Transform((r, c, v) => layerDerivatives[Layers.Length - 1](Inputs[Layers.Length - 1].GetColumn(j)[r]) * errors[r,j]);//errors.Transform((r, c, v) => layerDerivatives[Layers.Length - 1](Inputs[Layers.Length - 1][r, c]) * v);

                for (int i = Layers.Length - 2; i >= 0; i--)
                {
                    float Error(int row, int column, float value)
                    {
                        return PartialDerivatives[i + 1][row, column] * value;
                    }
                    float PartialD(int row, int column, float value)
                    {
                        float actPrime = layerDerivatives[i](Inputs[i][row, j]);

                        float error = Layers[i + 1].Item1.GetColumnAsMatrix(row).Transform(Error).Sum(a => a);
                        return actPrime * error;
                    }
                    PartialDerivatives[i] = Layers[i].Item1.Transform(PartialD);
                }

                //Calculate WeightUpdates
                WeightUpdates[0] = PartialDerivatives[0].Transform((r, c, v) => (r < inputs.Length && c < inputs[0].Length ?
                inputs[r][c] : 1)
                * learningRate * v);
                for (int i = 1; i < Layers.Length; i++)
                {
                    WeightUpdates[i] = PartialDerivatives[i].Transform((r, c, v) => (r < Outputs[i - 1].Rows && c < Outputs[i - 1].Columns ? Outputs[i - 1][r, c] : 1) * v * learningRate);
                }

                for (int i = 0; i < Layers.Length; i++)
                {
                    Layers[i].Item1 += WeightUpdates[i];
                }
            }

            return outputError;
        }

        private float[][] ComputeBatchWithRecords(float[][] inputs)
        {
            float[][] outputs = inputs;
            for (int i = 0; i < Layers.Length; i++)
            {
                (Inputs[i], outputs) = ComputeLayerBatchWithRecords(outputs, Layers[i].Item1, Layers[i].Item2);
                Outputs[i] = outputs;
            }

            return outputs;
        }

        private static (float[][], float[][]) ComputeLayerBatchWithRecords(float[][] inputs, Matrix layerWeights, Func<float, float> activationFunction)
        {
            float[][] insWithBias = new float[inputs.Length][];
            for (int i = 0; i < insWithBias.Length; i++)
            {
                insWithBias[i] = new float[inputs[i].Length + 1];
                insWithBias[i][0] = 1;
                inputs[i].CopyTo(insWithBias[i], 1);
            }
            var weightedInputs = layerWeights * insWithBias;
            var outputs = weightedInputs.Values.Select(x => x.Select(activationFunction));
            if (outputs is float[][] arr)
            {
                return (weightedInputs, arr);
            }

            var outs1 = outputs.ToArray();
            var outs2 = new float[outs1.Length][];
            for (var index = 0; index < outs1.Length; index++)
            {
                outs2[index] = outs1[index].ToArray();
            }

            return (weightedInputs, outs2);
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
