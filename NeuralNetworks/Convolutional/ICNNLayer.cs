using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.NeuralNetworks.Convolutional
{
    public interface ICNNLayer
    {
        float[][][] Compute(float[][][] input);
        int FilterSideLength { get; }
        int StrideLength { get; }
    }
}
