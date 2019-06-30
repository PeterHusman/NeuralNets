using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;

namespace NeuralNets.NeuralNetworks
{
    public class FeedForwardNeuralNetwork
    {
        //Columns = weights
        //Rows = neurons
        public (Matrix, Func<float, float>)[] Layers;

        public Matrix[] Inputs;
        public Matrix[] Outputs;
        public Matrix[] WeightUpdates;
        public float[][] PartialDerivatives;

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

        public static float[] ComputeLayer(float[] inputs, Matrix layerWeights, Func<float, float> activationFunction)
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
            ClearUpdates();
            //Matrix output = ComputeBatchWithRecords(inputs);
            //var errors = desiredOutputs - output;
            float outputError = 0f;// errors.Sum(a => Math.Abs(a)) / (errors.Columns * errors.Rows);//(desiredOutputs.Length * desiredOutputs[0].Length);
            if (float.IsInfinity(outputError))
            {
                ;
            }

            if (float.IsNaN(outputError))
            {
                ;
            }

            for (int j = 0; j < inputs.Length; j++)
            {
                Inputs = new Matrix[Layers.Length];
                Outputs = new Matrix[Layers.Length];
                Matrix outputs = (Matrix)ComputeBatchWithRecords(new[] { inputs[j] });

                var eachInput = Inputs.Select(a => a.Values[0]).ToArray();

                FindDerivatives(layerDerivatives, outputs.Values[0], desiredOutputs[j], eachInput, ref outputError);


                FindUpdates(inputs, learningRate);

            }

            outputError /= inputs.Length;

            UpdateWeights();

            return outputError;
        }

        private void FindUpdates(float[][] inputs, float learningRate)
        {
            //Calculate WeightUpdates
            WeightUpdates[0] += Layers[0].Item1.Transform((r, c, v) => (r < inputs.Length && c < inputs[0].Length ?
            inputs[r][c] : 1)
            * learningRate * PartialDerivatives[0][r]);
            for (int i = 1; i < Layers.Length; i++)
            {
                WeightUpdates[i] += Layers[i].Item1.Transform((r, c, v) => (r < Outputs[i - 1].Rows && c < Outputs[i - 1].Columns ? Outputs[i - 1][r, c] : 1) * PartialDerivatives[i][r] * learningRate);
            }
        }

        private void FindDerivatives(Func<float, float>[] layerDerivatives, float[] outputs, float[] targetOutputs, float[][] ins, ref float outputError)
        {
            float[] errors = new float[outputs.Length];

            float toAdd = 0;

            for(int i = 0; i < errors.Length; i++)
            {
                errors[i] = targetOutputs[i] - outputs[i];
                toAdd += Math.Abs(errors[i]);
            }
            toAdd /= errors.Length;
            outputError += toAdd;

            PartialDerivatives[Layers.Length - 1] = new float[Layers.Last().Item1.Rows];
            for (int i = 0; i < PartialDerivatives[Layers.Length - 1].Length; i++)
            {
                //Layers.Last().Item1.Transform((r, c, v) => layerDerivatives[Layers.Length - 1](Inputs[Layers.Length - 1].GetColumn(j)[r]) * errors[r,j]);//errors.Transform((r, c, v) => layerDerivatives[Layers.Length - 1](Inputs[Layers.Length - 1][r, c]) * v);
                PartialDerivatives[Layers.Length - 1][i] = layerDerivatives.Last()(ins[Layers.Length - 1][i]) * errors[i];
            }

            for (int i = Layers.Length - 2; i >= 0; i--)
            {
                float Error(int row, int column, float value)
                {
                    return /*row + 1 >= PartialDerivatives[i+1].Length - 1 ? 0 : */PartialDerivatives[i + 1][row] * value;
                }

                PartialDerivatives[i] = new float[Layers[i].Item1.Rows];
                for (int k = 0; k < PartialDerivatives[i].Length; k++)
                {
                    //= Layers[i].Item1.Transform(PartialD);
                    float actPrime = layerDerivatives[i](ins[i][k]);
                    float error = Layers[i + 1].Item1.GetColumnAsMatrix(k + 1)/*Added the plus-one to avoid including the bias*/.Transform(Error).Sum();
                    PartialDerivatives[i][k] = actPrime * error;
                }
            }
        }

        private void ClearUpdates()
        {
            Inputs = new Matrix[Layers.Length];
            Outputs = new Matrix[Layers.Length];
            WeightUpdates = new Matrix[Layers.Length];
            for (int i = 0; i < WeightUpdates.Length; i++)
            {
                WeightUpdates[i] = new Matrix(Layers[i].Item1.Rows, Layers[i].Item1.Columns);
            }
            PartialDerivatives = new float[Layers.Length][];
        }

        private void UpdateWeights()
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Item1 += WeightUpdates[i];
            }
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

        public static float[][] ComputeLayerBatch(float[][] inputs, Matrix layerWeights, Func<float, float> activationFunction)
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
