using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.NeuralNetworks
{
    public class ActivationFunction
    {

        public Func<double, double> Function;
        public Func<double, double> Derivative;

        public ActivationFunction(Func<double, double> func, Func<double, double> derivative)
        {
            Function = func;
            Derivative = derivative;
        }

        public static implicit operator Func<double, double>(ActivationFunction func)
        {
            return func.Function;
        }
    }
}
