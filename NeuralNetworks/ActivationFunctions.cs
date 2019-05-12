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
            get => new Dictionary<Func<float, float>, Func<float, float>> { [Sigmoid] = SigmoidDerivative, [ReLU] = ReLUDerivative, [BinaryStep] = BinaryStepDerivative, [SoftPlus] = SoftPlusDerivative };
        }

        public static float ReLU(float input)
        {
            return input < 0 ? 0 : input;
        }

        public static float ReLUDerivative(float input)
        {
            return input < 0 ? 0 : 1;
        }

        public static float BinaryStep(float input)
        {
            return input < 0 ? 0 : 1;
        }
        
        public static float BinaryStepDerivative(float input)
        {
            return 0;
        }

        public static float SoftPlus(float input)
        {
            return (float)Math.Log(1 + Math.Exp(input));
        }

        public static float SoftPlusDerivative(float input)
        {
            return 1 / (1 + (float)Math.Exp(-input));
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
