using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    public class Matrix
    {
        public float[][] Values { get; private set; }

        public float this[int row, int column]
        {
            get => Values[column][row];
            set => Values[column][row] = value;
        }

        public IEnumerable<float> GetRow(int rowNumber)
        {
            foreach (var column in Values)
            {
                yield return column[rowNumber];
            }
        }

        public float[] GetColumn(int columnNumber)
        {
            return Values[columnNumber];
        }

        public int Columns => Values.Length;

        public int Rows => Columns == 0 ? 0 : Values[0].Length;

        public Matrix(float[][] values)
        {
            Values = values;
        }

        public Matrix(int rows, int columns)
        {
            Values = new float[columns][];
            for (int i = 0; i < columns; i++)
            {
                Values[i] = new float[rows];
            }
        }

        public static Matrix operator *(Matrix left, Matrix right)
        {
            if (left.Columns != right.Rows)
            {
                throw new ArgumentException();
            }

            Matrix product = new Matrix(left.Rows, right.Columns);

            for (int r = 0; r < product.Rows; r++)
            {
                for (int c = 0; c < product.Columns; c++)
                {
                    product[r, c] = VectorDotProduct(right.GetColumn(c), left.GetRow(r));
                }
            }

            return product;
        }

        public static Matrix operator +(Matrix left, Matrix right)
        {

        }

        public static float VectorDotProduct(IEnumerable<float> left, IEnumerable<float> right)
        {
            float output = 0f;
            IEnumerator<float> leftEnumerator = left.GetEnumerator();
            IEnumerator<float> rightEnumerator = right.GetEnumerator();
            do
            {
                output += leftEnumerator.Current * rightEnumerator.Current;
            } while (leftEnumerator.MoveNext() && rightEnumerator.MoveNext());
            leftEnumerator.Dispose();
            rightEnumerator.Dispose();
            return output;
        }

        public static IEnumerable<float> VectorSum(IEnumerable<float> left, IEnumerable<float> right)
        {

        }
    }
}
