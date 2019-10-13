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

        public float[] Biases { get; private set; }

        public int OutputSideLength { get; private set; }

        public int ExpectedInputDepth { get; private set; }

        public int ExpectedInputWidth { get; private set; }

        public bool UseReLU { get; private set; }

        public float[][][] Compute(float[][][] input)
        {
            //i is layer in output
            LastOuts = new float[Depth][][];
            for(int i = 0; i < Depth; i++)
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
                                if(absY <= ZeroPaddingSize || absY >= ExpectedInputWidth + ZeroPaddingSize)
                                {
                                    continue;
                                }
                                //l is part of position in filter
                                for (int l = 0; l < FilterSideLength; l++)
                                {
                                    int absX = x * StrideLength + l;
                                    if (absX <= ZeroPaddingSize || absX >= ExpectedInputWidth + ZeroPaddingSize)
                                    {
                                        continue;
                                    }
                                    LastOuts[i][y][x] += input[j][absY][absX] * Weights[i][j][k][l];
                                }
                            }
                        }
                        LastOuts[i][y][x] += Biases[i];
                        if(UseReLU && LastOuts[i][y][x] < 0)
                        {
                            LastOuts[i][y][x] = 0;
                        }
                    }
                }
            }
            return LastOuts;
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
            if(d != (double)OutputSideLength)
            {
                throw new InvalidOperationException();
            }
            Weights = new float[depth][][][];
            for(int i = 0; i < depth; i++)
            {
                Weights[i] = new float[inputDepth][][];
                for(int j = 0; j < inputDepth; j++)
                {
                    Weights[i][j] = new float[filterSize][];
                    for(int k = 0; k < filterSize; k++)
                    {
                        Weights[i][j][k] = new float[filterSize];
                    }
                }
            }
        }
    }
}
