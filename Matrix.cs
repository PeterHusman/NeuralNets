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
        public static implicit operator Matrix(float[][] a)
        {
            return new Matrix(a);
        }

        public static implicit operator float[][](Matrix a)
        {
            return a.Values;
        }

        protected bool Equals(Matrix other)
        {
            return this == other;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((Matrix) obj);
        }

        public override int GetHashCode()
        {
            return (Values != null ? Values.GetHashCode() : 0);
        }

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

        public Matrix GetColumnAsMatrix(int columnNumber)
        {
            return new Matrix(new[] { Values[columnNumber] });
        }

        public Matrix GetRowAsMatrix(int rowNumber)
        {
            float[][] values = new float[Columns][];
            for(int i = 0; i < Columns; i++)
            {
                values[i] = new[] { Values[i][rowNumber] };
            }
            return new Matrix(values);
        }

        public void TransformValues(Func<int, int, float, float> func)
        {
            for(int r = 0; r < Rows; r++)
            {
                for(int c = 0; c < Columns; c++)
                {
                    this[r, c] = func(r, c, this[r, c]);
                }
            }
        }

        public Matrix Transform(Func<int, int, float, float> func)
        {
            Matrix output = new Matrix(Rows, Columns);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    output[r, c] = func(r, c, this[r, c]);
                }
            }
            return output;
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

        public Matrix GetTranspose()
        {
            Matrix outputMatrix = new Matrix(Columns, Rows);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    outputMatrix[c, r] = this[r, c];
                }
            }

            return outputMatrix;
        }

        public static bool operator ==(Matrix left, Matrix right)
        {
            if (left == null || right == null || left.Columns != right.Columns || left.Rows != right.Rows)
            {
                return false;
            }

            for (int i = 0; i < left.Columns; i++)
            {
                for (int j = 0; j < left.Rows; j++)
                {
                    if (left[j, i] != right[j,i])
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        public static bool operator !=(Matrix left, Matrix right)
        {
            return !(left == right);
        }

        public static Matrix operator *(float left, Matrix right)
        {
            Matrix output = new Matrix(right.Rows, right.Columns);
            for(int i = 0; i < right.Values.Length; i++)
            {
                for(int j = 0; j < right.Values[i].Length; j++)
                {
                    output.Values[i][j] = right.Values[i][j] * left;
                }
            }
            return output;
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
            Matrix output = new Matrix(left.Rows, left.Columns);
            for (int i = 0; i < output.Columns; i++)
            {
                output.Values[i] = VectorSum(left.Values[i], right.Values[i]).ToArray();
            }

            return output;
        }

        public static Matrix operator -(Matrix matrix)
        {
            Matrix output = new Matrix(matrix.Values);
            for (int r = 0; r < matrix.Rows; r++)
            {
                for (int c = 0; c < matrix.Columns; c++)
                {
                    output[r, c] *= -1;
                }
            }

            return output;
        }

        public static Matrix operator -(Matrix left, Matrix right)
        {
            Matrix output = new Matrix(left.Rows, left.Columns);
            for (int i = 0; i < output.Columns; i++)
            {
                output.Values[i] = VectorSum(left.Values[i], right.Values[i].Select(x => -x)).ToArray();
            }

            return output;
        }

        public int NumberOfItems()
        {
            return Columns * Rows;
        }

        public float Sum(Func<float, float> accumulator)
        {
            float sum = 0;
            for (int i = 0; i < Columns; i++)
            {
                for (int j = 0; j < Rows; j++)
                {
                    sum += accumulator(this[j, i]);
                }
            }

            return sum;
        }

        public static float VectorDotProduct(IEnumerable<float> left, IEnumerable<float> right)
        {
            float output = 0f;
            IEnumerator<float> leftEnumerator = left.GetEnumerator();
            IEnumerator<float> rightEnumerator = right.GetEnumerator();
            while(leftEnumerator.MoveNext() && rightEnumerator.MoveNext())
            {
                output += leftEnumerator.Current * rightEnumerator.Current;
            }
            leftEnumerator.Dispose();
            rightEnumerator.Dispose();
            return output;
        }

        public static IEnumerable<float> VectorSum(IEnumerable<float> left, IEnumerable<float> right)
        {
            IEnumerator<float> leftEnumerator = left.GetEnumerator();
            IEnumerator<float> rightEnumerator = right.GetEnumerator();
            while (leftEnumerator.MoveNext() && rightEnumerator.MoveNext())
            {
                yield return leftEnumerator.Current + rightEnumerator.Current;
            }
            leftEnumerator.Dispose();
            rightEnumerator.Dispose();
        }
    }
}
