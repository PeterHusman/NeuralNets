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

        public float[][][] LastIns { get; private set; }

        public float[] Biases { get; private set; }

        public int OutputSideLength { get; private set; }

        public int ExpectedInputDepth { get; private set; }

        public int ExpectedInputWidth { get; private set; }

        public bool UseReLU { get; private set; }

        public float[][][] Compute(float[][][] input)
        {
            LastIns = input;
            //i is layer in output
            LastOuts = new float[Depth][][];
            for (int i = 0; i < Depth; i++)
            {
                //y is part of position in output
                LastOuts[i] = new float[OutputSideLength][];
                for (int y = 0; y < OutputSideLength; y++)
                {
                    //x is part of position in output
                    LastOuts[i][y] = new float[OutputSideLength];
                    for (int x = 0; x < OutputSideLength; x++)
                    {
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
                                    LastOuts[i][y][x] += input[j][absY][absX] * Weights[i][j][k][l];
                                }
                            }
                        }
                        LastOuts[i][y][x] += Biases[i];
                        if (UseReLU && LastOuts[i][y][x] < 0)
                        {
                            LastOuts[i][y][x] = 0;
                        }
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
                            Weights[i][j][k][l] -= learningRate * dErrorDWeight[i][j][k][l];
                        }
                    }
                }
                Biases[i] -= biasChanges[i] * learningRate;
            }
        }

        private float[][][] dErrorDInput { get; set; }
        private float[][][][] dErrorDWeight { get; set; }
        private float[] biasChanges { get; set; }

        public void ClearUpdates()
        {
            dErrorDInput = new float[ExpectedInputDepth][][];
            dErrorDWeight = new float[Depth][][][];
            biasChanges = new float[Depth];
            for (int i = 0; i < ExpectedInputDepth; i++)
            {
                dErrorDInput[i] = new float[ExpectedInputWidth][];
                for (int j = 0; j < ExpectedInputWidth; j++)
                {
                    dErrorDInput[i][j] = new float[ExpectedInputWidth];
                }
            }
            for (int i = 0; i < Depth; i++)
            {
                dErrorDWeight[i] = new float[ExpectedInputDepth][][];
                for (int j = 0; j < ExpectedInputDepth; j++)
                {
                    dErrorDWeight[i][j] = new float[FilterSideLength][];
                    for (int k = 0; k < FilterSideLength; k++)
                    {
                        dErrorDWeight[i][j][k] = new float[FilterSideLength];
                    }
                }
            }

        }

        public float[][][] BackPropagation(float[][][] derivatives)
        {
            ClearUpdates();
            
            //i is depth in output
            for (int i = 0; i < Depth; i++)
            {
                //y is part of position in output
                for (int y = 0; y < OutputSideLength; y++)
                {
                    //x is part of position in output
                    for (int x = 0; x < OutputSideLength; x++)
                    {
                        if (UseReLU && LastOuts[i][y][x] == 0)
                        {
                            continue;
                        }
                        biasChanges[i] += derivatives[i][y][x];
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
                                    dErrorDInput[j][absY][absX] += Weights[i][j][k][l] * derivatives[i][y][x];
                                    dErrorDWeight[i][j][k][l] += derivatives[i][y][x] * LastIns[j][absY][absX];
                                }
                            }
                        }
                    }
                }
            }

            

            //TODO: Bias gradient descent.
            //Also TODO: Test.

            //https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e

            return dErrorDInput;
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

        public ConvolutionalLayer(int inputWidth, int filterSize, int padding, int stride, int depth, int inputDepth, bool useReLU)
        {
            ExpectedInputWidth = inputWidth;
            ExpectedInputDepth = inputDepth;
            FilterSideLength = filterSize;
            ZeroPaddingSize = padding;
            StrideLength = stride;
            Depth = depth;
            UseReLU = useReLU;
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
