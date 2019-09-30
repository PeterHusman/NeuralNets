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
            get => new Dictionary<Func<float, float>, Func<float, float>>();// { [Sigmoid] = SigmoidDerivative, [ReLU] = ReLUDerivative, [BinaryStep] = BinaryStepDerivative, [SoftPlus] = SoftPlusDerivative, [Identity] = IdentityDerivative };
        }

        public static ActivationFunction TanH = new ActivationFunction(Math.Tanh, TanHDerivative);

        public static ActivationFunction ReLU = new ActivationFunction(ReLUFunc, ReLUDerivative);

        public static ActivationFunction BinaryStep = new ActivationFunction(BinaryStepFunc, BinaryStepDerivative);

        public static ActivationFunction SoftPlus = new ActivationFunction(SoftPlusFunc, SoftPlusDerivative);

        public static ActivationFunction Sigmoid = new ActivationFunction(SigmoidFunc, SigmoidDerivative);

        public static ActivationFunction Identity = new ActivationFunction(IdentityFunc, IdentityDerivative);

        public static ActivationFunction LeakyReLU = new ActivationFunction(LeakyReLUFunc, LeakyReLUDerivative);

        private static double TanHDerivative(double input)
        {
            double d = Math.Tanh(input);
            return 1 - (d * d);
        }

        private static double ReLUFunc(double input)
        {
            return input < 0 ? 0 : input;
        }

        private static double ReLUDerivative(double input)
        {
            return input < 0 ? 0 : 1;
        }

        private static double LeakyReLUFunc(double input)
        {
            return input < 0 ? input * 0.01 : input;
        }

        private static double LeakyReLUDerivative(double input)
        {
            return input < 0 ? 0.01 : 1;
        }

        private static double BinaryStepFunc(double input)
        {
            return input < 0 ? 0 : 1;
        }
        
        private static double BinaryStepDerivative(double input)
        {
            return 0;
        }

        private static double SoftPlusFunc(double input)
        {
            return Math.Log(1 + Math.Exp(input));
        }

        private static double SoftPlusDerivative(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

        private static double SigmoidFunc(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

        private static double SigmoidDerivative(double input)
        {
            double output = SigmoidFunc(input);
            return output * (1 - output);
        }

        private static double IdentityFunc(double input)
        {
            return input;
        }

        private static double IdentityDerivative(double input)
        {
            return 1;
        }

    }
}
