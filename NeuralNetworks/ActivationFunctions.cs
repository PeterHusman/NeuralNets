using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.NeuralNetworks
{
    public static class ActivationFunctions
    {
        public static Dictionary<Func<float, float>, Func<float, float>> DefaultFunctionData
        {
            get => new Dictionary<Func<float, float>, Func<float, float>> { [Sigmoid] = SigmoidDerivative };
        }

        public static float Sigmoid(float input)
        {
            return 1 / (1 + (float)Math.Exp(-input));
        }

        public static float SigmoidDerivative(float input)
        {
            float output = Sigmoid(input);
            return output * (1 - output);
        }

    }
}
