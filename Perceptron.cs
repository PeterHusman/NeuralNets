using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace NeuralNets
{
    public static class Extensions
    {
        public static float[] ToFloats(this string[] strings)
        {
            float[] output = new float[strings.Length];
            for (int i = 0; i < strings.Length; i++)
            {
                output[i] = float.Parse(strings[i]);
            }

            return output;
        }
    }
    public class Perceptron
    {
        public Matrix WeightMatrix { get; private set; }

        public Perceptron(float[] weights)
        {
            WeightMatrix = new Matrix(1, weights.Length);
            for (int i = 0; i < weights.Length; i++)
            {
                WeightMatrix[0, i] = weights[i];
            }
        }

        public Perceptron()
        {
            WeightMatrix = new Matrix(0,0);
        }


        public void UpdateWeights(float[] weights)
        {
            WeightMatrix = new Matrix(1, weights.Length);
            for (int i = 0; i < weights.Length; i++)
            {
                WeightMatrix[0, i] = weights[i];
            }
        }

        public float Compute(float[] inputs)
        {
            Matrix valsMatrix = new Matrix(inputs.Length + 1, 1) {[0, 0] = 1};
            for (int i = 0; i < inputs.Length; i++)
            {
                valsMatrix[i + 1, 0] = inputs[i];
            }
            return (WeightMatrix * valsMatrix)[0,0];
        }
    }
}
