using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.NeuralNetworks.Convolutional
{
    public class PoolingLayer : ICNNLayer
    {
        public int FilterSideLength { get; private set; }

        public int StrideLength { get; private set; }

        public float[][][] LastOuts { get; private set; }
        public float[][][] LastIns { get; private set; }

        public int ExpectedInputWidth { get; private set; }

        public int OutputSideLength { get; private set; }

        public int ExpectedInputDepth { get; private set; }

        public int ZeroPaddingSize { get; private set; }

        public float[][][] Compute(float[][][] input)
        {
            LastIns = input;
            //i is layer in input and output
            LastOuts = new float[ExpectedInputDepth][][];
            for (int i = 0; i < ExpectedInputDepth; i++)
            {
                //y is part of position in output
                LastOuts[i] = new float[OutputSideLength][];
                for (int y = 0; y < OutputSideLength; y++)
                {
                    //x is part of position in output
                    LastOuts[i][y] = new float[OutputSideLength];
                    for (int x = 0; x < OutputSideLength; x++)
                    {
                        LastOuts[i][y][x] = float.NegativeInfinity;
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
                                if (input[i][absY][absX] > LastOuts[i][y][x])
                                {
                                    LastOuts[i][y][x] = input[i][absY][absX];
                                }
                            }
                        }
                    }
                }
            }
            return LastOuts;
        }

        public int Depth => ExpectedInputDepth;

        public float[][][] BackPropagation(float[][][] errors)
        {
            throw new NotImplementedException();
        }

        public float[][][] BackPropagation(float[][][] errors, ICNNLayer nextLayer)
        {
            float[][][] PartialDerivative = new float[Depth][][];
            for (int i = 0; i < Depth; i++)
            {
                PartialDerivative[i] = new float[OutputSideLength][];
                for (int j = 0; j < OutputSideLength; j++)
                {
                    PartialDerivative[i][j] = new float[OutputSideLength];
                }
            }

            ConvolutionalLayer conv = nextLayer as ConvolutionalLayer;

            //i is layer in input and output
            for (int i = 0; i < nextLayer.Depth; i++)
            {
                //y is part of position in output
                for (int y = 0; y < nextLayer.OutputSideLength; y++)
                {
                    //x is part of position in output
                    for (int x = 0; x < nextLayer.OutputSideLength; x++)
                    {
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
                                    if (conv != null)
                                    {
                                        weightThing = conv.Weights[i][j][k][l];
                                    }
                                    else
                                    {
                                        throw new NotImplementedException();
                                    }
                                    PartialDerivative[j][absY][absX] += weightThing * errors[j][y][x];

                                }
                            }
                        }
                    }
                }
            }

            return PartialDerivative;
        }

        void ICNNLayer.Randomize(Random random) { }

        void ICNNLayer.ClearUpdates()
        {
        }

        void ICNNLayer.ApplyUpdates(float learningRate)
        {
        }

        public PoolingLayer(int inputWidth, int filterSize, int padding, int stride, int inputDepth)
        {
            ExpectedInputWidth = inputWidth;
            ExpectedInputDepth = inputDepth;
            FilterSideLength = filterSize;
            ZeroPaddingSize = padding;
            StrideLength = stride;
            double d = ((double)inputWidth - filterSize + 2 * padding) / stride + 1;
            OutputSideLength = (int)d;
            if (d != (double)OutputSideLength)
            {
                throw new ArgumentException();
            }
        }
    }
}
