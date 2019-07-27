using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.NeuralNetworks
{
    public class FeedForwardNetwork
    {
        public Layer[] Layers;
        public double[] Output;

        public FeedForwardNetwork(int inputCount, params (int, ActivationFunction)[] layerInfos)
        {
            Layers = new Layer[layerInfos.Length];

            Layers[0] = new Layer(layerInfos[0].Item2, inputCount, layerInfos[0].Item1);

            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(layerInfos[i].Item2, layerInfos[i - 1].Item1, layerInfos[i].Item1);
            }
        }

        public void Randomize(Random rand)
        {
            for(int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Randomize(rand);
            }
        }

        public double[] Compute(double[] input)
        {
            Output = input;
            for(int i = 0; i < Layers.Length; i++)
            {
                Output = Layers[i].Compute(Output);
            }

            return Output;
        }

        public double MAE(double[] inputs, double[] desiredOutputs)
        {
            double error = 0;
            double[] outs = Compute(inputs);
            
            for(int i = 0; i < outs.Length; i++)
            {
                error += Math.Abs(outs[i] - desiredOutputs[i]);
            }

            error /= outs.Length;

            return error;
        }

        public double GradientDescent(double[][] inputs, double[][] desiredOutputs, double learningRate, double momentum, out double[][] actualOutputs)
        {
            double totalError = 0;

            actualOutputs = new double[desiredOutputs.Length][];

            foreach(Layer layer in Layers)
            {
                foreach (Neuron neuron in layer.Neurons)
                {
                    neuron.WeightUpdates = new double[neuron.Weights.Length];
                    neuron.BiasUpdate = 0;
                }
            }

            for(int i = 0; i < inputs.Length; i++)
            {
                Compute(inputs[i]);

                actualOutputs[i] = new double[desiredOutputs.Length];

                Layer outputLayer = Layers[Layers.Length - 1];

                for(int j = 0; j < outputLayer.Neurons.Length; j++)
                {
                    Neuron neuron = outputLayer.Neurons[j];

                    actualOutputs[i][j] = neuron.Output;

                    double error = desiredOutputs[i][j] - neuron.Output;

                    totalError += error * error;

                    neuron.PartialDerivative = error * neuron.ActivationFunction.Derivative(neuron.Input);
                }

                for(int j = Layers.Length - 2; j >= 0; j--)
                {
                    Layer layer = Layers[j];
                    for(int k = 0; k < layer.Neurons.Length; k++)
                    {
                        Neuron neuron = layer.Neurons[k];

                        double error = 0;

                        foreach(Neuron nextNeuron in Layers[j + 1].Neurons)
                        {
                            error += nextNeuron.PartialDerivative * nextNeuron.Weights[k];
                        }

                        neuron.PartialDerivative = error * neuron.ActivationFunction.Derivative(neuron.Input);
                    }
                }

                Layer firstLayer = Layers[0];

                for (int j = 0; j < firstLayer.Neurons.Length; j++)
                {
                    Neuron neuron = firstLayer.Neurons[j];
                    for(int k = 0; k < neuron.Weights.Length; k++)
                    {
                        neuron.WeightUpdates[k] += learningRate * neuron.PartialDerivative * inputs[i][k];
                    }

                    neuron.BiasUpdate += learningRate * neuron.PartialDerivative;
                }

                for(int j = 1; j < Layers.Length; j++)
                {
                    Layer layer = Layers[j];
                    Layer prevLayer = Layers[j - 1];

                    for (int k = 0; k < layer.Neurons.Length; k++)
                    {
                        Neuron neuron = layer.Neurons[k];
                        for (int l = 0; l < neuron.Weights.Length; l++)
                        {
                            neuron.WeightUpdates[l] += learningRate * neuron.PartialDerivative * prevLayer.Output[l];
                        }

                        neuron.BiasUpdate += learningRate * neuron.PartialDerivative;
                    }
                }

            }

            for(int i = 0; i < Layers.Length; i++)
            {
                for(int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    for(int k = 0; k < Layers[i].Neurons[j].Weights.Length; k++)
                    {
                        double update = Layers[i].Neurons[j].WeightUpdates[k] + (momentum * Layers[i].Neurons[j].PrevWeightUpdates[k]);
                        Layers[i].Neurons[j].PrevWeightUpdates[k] = update;
                        Layers[i].Neurons[j].Weights[k] += update;
                    }
                    double upd = Layers[i].Neurons[j].BiasUpdate + (momentum * Layers[i].Neurons[j].PrevBiasUpdate);
                    Layers[i].Neurons[j].Bias += upd;
                    Layers[i].Neurons[j].PrevBiasUpdate = upd;
                }
            }

            return totalError / inputs.Length;
        }
    }
}
