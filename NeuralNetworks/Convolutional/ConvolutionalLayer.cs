using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.NeuralNetworks.Convolutional
{
    public class ConvolutionalLayer : ICNNLayer
    {
        public int FilterSideLength { get; private set; }

        public int StrideLength { get; private set; }

        public int Depth { get; private set; }

        public int ZeroPaddingSize { get; private set; }

        public float[][][][] Weights { get; private set; }

        public float[][][] LastOuts { get; private set; }

        public float[][][] ReallyLastIns { get; private set; }

        public float[] Biases { get; private set; }

        public int OutputSideLength { get; private set; }

        public int ExpectedInputDepth { get; private set; }

        public int ExpectedInputWidth { get; private set; }

        public ActivationFunction ActivationFunc { get; private set; }

        public float[][][] Compute(float[][][] input)
        {
            LastIns = new float[ExpectedInputDepth][][];
            ReallyLastIns = new float[Depth][][];
            //i is layer in output
            LastOuts = new float[Depth][][];
            for (int i = 0; i < Depth; i++)
            {
                //y is part of position in output
                LastOuts[i] = new float[OutputSideLength][];
                ReallyLastIns[i] = new float[OutputSideLength][];
                for (int y = 0; y < OutputSideLength; y++)
                {
                    //x is part of position in output
                    LastOuts[i][y] = new float[OutputSideLength];
                    ReallyLastIns[i][y] = new float[OutputSideLength];
                    for (int x = 0; x < OutputSideLength; x++)
                    {
                        //j is depth in input
                        for (int j = 0; j < ExpectedInputDepth; j++)
                        {
                            if (LastIns[j] == null)
                            {
                                LastIns[j] = new float[ExpectedInputWidth][];
                                for (int abc = 0; abc < ExpectedInputWidth; abc++)
                                {
                                    LastIns[j][abc] = new float[ExpectedInputWidth];
                                    for (int cba = 0; cba < ExpectedInputWidth; cba++)
                                    {
                                        LastIns[j][abc][cba] = input[j][abc][cba];
                                    }
                                }
                            }
                            //k is part of position in filter
                            for (int k = 0; k < FilterSideLength; k++)
                            {
                                int absY = y * StrideLength + k;
                                if (absY < ZeroPaddingSize || absY >= ExpectedInputWidth + ZeroPaddingSize)
                                {
                                    continue;
                                }
                                //l is part of position in filter
                                for (int l = 0; l < FilterSideLength; l++)
                                {
                                    int absX = x * StrideLength + l;
                                    if (absX < ZeroPaddingSize || absX >= ExpectedInputWidth + ZeroPaddingSize)
                                    {
                                        continue;
                                    }
                                    ReallyLastIns[i][y][x] += input[j][absY][absX] * Weights[i][j][k][l];
                                }
                            }
                        }
                        ReallyLastIns[i][y][x] += Biases[i];
                        LastOuts[i][y][x] = (float)ActivationFunc.Function(ReallyLastIns[i][y][x]);
                    }
                }
            }
            return LastOuts;
        }

        public void ApplyUpdates(float learningRate)
        {
            for (int i = 0; i < Depth; i++)
            {
                for (int j = 0; j < ExpectedInputDepth; j++)
                {
                    for (int k = 0; k < FilterSideLength; k++)
                    {
                        for (int l = 0; l < FilterSideLength; l++)
                        {
                            Weights[i][j][k][l] -= learningRate * DErrorDWeight[i][j][k][l];
                        }
                    }
                }
                Biases[i] -= BiasChanges[i] * learningRate;
            }
        }

        public float[][][] PartialDerivative { get; set; }
        public float[][][][] DErrorDWeight { get; set; }
        public float[] BiasChanges { get; set; }
        public float[][][] LastIns { get; set; }

        public void ClearUpdates()
        {
            PartialDerivative = new float[Depth][][];
            DErrorDWeight = new float[Depth][][][];
            BiasChanges = new float[Depth];
            for (int i = 0; i < Depth; i++)
            {
                PartialDerivative[i] = new float[OutputSideLength][];
                for (int j = 0; j < OutputSideLength; j++)
                {
                    PartialDerivative[i][j] = new float[OutputSideLength];
                }
            }
            for (int i = 0; i < Depth; i++)
            {
                DErrorDWeight[i] = new float[ExpectedInputDepth][][];
                for (int j = 0; j < ExpectedInputDepth; j++)
                {
                    DErrorDWeight[i][j] = new float[FilterSideLength][];
                    for (int k = 0; k < FilterSideLength; k++)
                    {
                        DErrorDWeight[i][j][k] = new float[FilterSideLength];
                    }
                }
            }

        }

        public float[][][] BackPropagation(float[][][] derivatives, ICNNLayer nextLayer)
        {
            ConvolutionalLayer conv = nextLayer as ConvolutionalLayer;
            //i is depth in output
            for (int i = 0; i < nextLayer.Depth; i++)
            {
                //y is part of position in output
                for (int y = 0; y < nextLayer.OutputSideLength; y++)
                {
                    //x is part of position in output
                    for (int x = 0; x < nextLayer.OutputSideLength; x++)
                    {
                        //j is depth in input
                        for (int j = 0; j < nextLayer.ExpectedInputDepth; j++)
                        {
                            //k is part of position in filter
                            for (int k = 0; k < nextLayer.FilterSideLength; k++)
                            {
                                int absY = y * nextLayer.StrideLength + k;
                                if (absY < nextLayer.ZeroPaddingSize || absY >= nextLayer.ExpectedInputWidth + nextLayer.ZeroPaddingSize)
                                {
                                    continue;
                                }
                                //l is part of position in filter
                                for (int l = 0; l < nextLayer.FilterSideLength; l++)
                                {
                                    int absX = x * nextLayer.StrideLength + l;
                                    if (absX < nextLayer.ZeroPaddingSize || absX >= nextLayer.ExpectedInputWidth + nextLayer.ZeroPaddingSize)
                                    {
                                        continue;
                                    }

                                    float weightThing = 1f;
                                    if (conv == null)
                                    {
                                        weightThing = nextLayer.LastIns[j][absY][absX] == nextLayer.LastOuts[i][y][x] ? 1 : 0;
                                    }
                                    else
                                    {
                                        weightThing = conv.Weights[i][j][k][l];
                                    }
                                    PartialDerivative[j][absY][absX] += weightThing * derivatives[j][y][x];
                                }
                            }
                        }
                    }
                }
            }
            return Backprop(derivatives);
        }

        public float[][][] BackPropagation(float[][][] derivatives)
        {
            //i is depth in output
            for (int i = 0; i < Depth; i++)
            {
                //y is part of position in output
                for (int y = 0; y < OutputSideLength; y++)
                {
                    //x is part of position in output
                    for (int x = 0; x < OutputSideLength; x++)
                    {
                        /*//j is depth in input
                        for (int j = 0; j < ExpectedInputDepth; j++)
                        {
                            //k is part of position in filter
                            for (int k = 0; k < FilterSideLength; k++)
                            {
                                int absY = y * StrideLength + k;
                                if (absY < ZeroPaddingSize || absY >= ExpectedInputWidth + ZeroPaddingSize)
                                {
                                    continue;
                                }
                                //l is part of position in filter
                                for (int l = 0; l < FilterSideLength; l++)
                                {
                                    int absX = x * StrideLength + l;
                                    if (absX < ZeroPaddingSize || absX >= ExpectedInputWidth + ZeroPaddingSize)
                                    {
                                        continue;
                                    }
                                    */
                        PartialDerivative[i][y][x] += derivatives[i][y][x];
                        /*}
                    }
                }*/
                    }
                }
            }
            return Backprop(derivatives);
        }

        public float[][][] Backprop(float[][][] derivatives)
        {
            //i is depth in output
            for (int i = 0; i < Depth; i++)
            {
                //y is part of position in output
                for (int y = 0; y < OutputSideLength; y++)
                {
                    //x is part of position in output
                    for (int x = 0; x < OutputSideLength; x++)
                    {
                        PartialDerivative[i][y][x] *= (float)ActivationFunc.Derivative(ReallyLastIns[i][y][x]);
                        BiasChanges[i] += PartialDerivative[i][y][x];
                        //j is depth in input
                        for (int j = 0; j < ExpectedInputDepth; j++)
                        {
                            //k is part of position in filter
                            for (int k = 0; k < FilterSideLength; k++)
                            {
                                int absY = y * StrideLength + k;
                                if (absY < ZeroPaddingSize || absY >= ExpectedInputWidth + ZeroPaddingSize)
                                {
                                    continue;
                                }
                                //l is part of position in filter
                                for (int l = 0; l < FilterSideLength; l++)
                                {
                                    int absX = x * StrideLength + l;
                                    if (absX < ZeroPaddingSize || absX >= ExpectedInputWidth + ZeroPaddingSize)
                                    {
                                        continue;
                                    }
                                    DErrorDWeight[i][j][k][l] += PartialDerivative[j][y][x] * LastIns[j][absY][absX];
                                }
                            }
                        }
                    }
                }
            }


            //TODO: Bias gradient descent.
            //Also TODO: Test.

            //https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e

            return PartialDerivative;
        }

        public void Randomize(Random random)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                for (int j = 0; j < Weights[i].Length; j++)
                {
                    for (int k = 0; k < Weights[i][j].Length; k++)
                    {
                        for (int l = 0; l < Weights[i][j][k].Length; l++)
                        {
                            Weights[i][j][k][l] = (float)random.NextDouble() * 2 - 1;
                        }
                    }
                }
            }

            for (int i = 0; i < Depth; i++)
            {
                Biases[i] = (float)random.NextDouble() * 2 - 1;
            }
        }

        public ConvolutionalLayer(int inputWidth, int filterSize, int padding, int stride, int depth, int inputDepth, ActivationFunction activationFunc)
        {
            ExpectedInputWidth = inputWidth;
            ExpectedInputDepth = inputDepth;
            FilterSideLength = filterSize;
            ZeroPaddingSize = padding;
            StrideLength = stride;
            Depth = depth;
            ActivationFunc = activationFunc;
            double d = ((double)inputWidth - filterSize + 2 * padding) / stride + 1;
            OutputSideLength = (int)d;
            if (d != (double)OutputSideLength)
            {
                throw new ArgumentException();
            }
            Biases = new float[depth];
            Weights = new float[depth][][][];
            for (int i = 0; i < depth; i++)
            {
                Weights[i] = new float[inputDepth][][];
                for (int j = 0; j < inputDepth; j++)
                {
                    Weights[i][j] = new float[filterSize][];
                    for (int k = 0; k < filterSize; k++)
                    {
                        Weights[i][j][k] = new float[filterSize];
                    }
                }
            }
        }
    }
}
