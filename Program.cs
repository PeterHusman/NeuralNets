using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NeuralNets.NeuralNetworks;

namespace NeuralNets
{
    internal class Program
    {
        static Func<float, float> act = a => a;
        static Func<float, float> relu = a => a > 0 ? a : 0;
        static Func<float, float> bin = a => a > 0 ? 1 : 0;
        private static async Task Main(string[] args)
        {
            int timeWasted = 0;
            string[] timeWastedStrings = { "Please make a selection.", "Really? I don't have all day.", "Seriously?!?", "I've had enough..." };
            while (true)
            {
                CancellationTokenSource source = new CancellationTokenSource();
                CancellationToken token = source.Token;
                var waitTask = Task.Delay(5000, token);
                int selection = -1;

                Task<int> chooseTask = Task.Run(() => CHelper.SelectorMenu(@"Please select the program to run.", new[] { "Hill Climber", "Perceptron", "XORNet - Random Train", "Gradient Descent" },
                    true, ConsoleColor.DarkYellow, ConsoleColor.Gray, ConsoleColor.Magenta, token), token);
                //waitTask.Start();
                //chooseTask.Start();
                if (await Task.WhenAny(waitTask, chooseTask) == chooseTask)
                {
                    selection = chooseTask.Result;
                    //chooseTask.Dispose();
                }
                else
                {

                }
                //else
                //{
                source.Cancel();
                //}
                source.Dispose();
                switch (selection)
                {
                    case -1:
                        Console.Clear();
                        await CHelper.SlowWriteLine(ConsoleColor.DarkYellow, timeWastedStrings[timeWasted], 50);
                        timeWasted++;
                        if (timeWasted >= timeWastedStrings.Length)
                        {
                            throw new Exception();
                        }
                        break;
                    case 0:
                        await HillClimber();
                        break;
                    case 1:
                        PerceptronTest();
                        break;
                    case 2:
                        await XORNetTest();
                        break;
                    case 3:
                        await GradientTest();
                        break;
                }
                //waitTask.Dispose();
                //chooseTask.Dispose();
                if (selection != -1 && timeWasted > 0)
                {
                    await CHelper.SlowWriteLine(ConsoleColor.DarkYellow, "Thank you for choosing promptly.", 50);
                }
            }
        }

        private static async Task GradientTest()
        {
            Random rand = new Random();
            var net = NeuralNetworkFactory.CreateRandomizedFeedForwardNeuralNetwork(rand, 2, (2, ActivationFunctions.Sigmoid), (2, ActivationFunctions.Sigmoid), (1, ActivationFunctions.Sigmoid));
            int n = 50;
            float[][] inputs = new[] { /*new[] { 0f, 0f }, new[] { 0f, 1f },*/ new[] { 1f, 0f, }, new[] { 1f, 1f } };//new float[n][];
            float[][] outputs = new[] { /*new[] { 0f }, new[] { 1f },*/ new[] { 1f }, new[] { 0f } };//new float[n][];
            /*for(int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = new float[] { i };
                outputs[i] = new float[] { 2*i };
            }*/
            var a = NeuralNetworkFactory.GradientDescentTrainCoroutine(net, inputs, outputs, 0.01f, 0.01f);

            foreach (var b in a)
            {
                Matrix input = new[] { new[] { 1f, 0f } };
                float target = input.Values[0][0] == input.Values[0][1] ? 0 : 1;
                Console.SetCursorPosition(0, 5);
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine(b);
                Console.WriteLine();
                Matrix output = input;
                Console.WriteLine(output);
                output = FeedForwardNeuralNetwork.ComputeLayerBatch(new[] { new float[] { 1f, 0f } }, net.Layers[0].Item1, net.Layers[0].Item2);
                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.WriteLine(new Matrix(new[] { net.PartialDerivatives[0] }));
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine(output.Stringify);
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.WriteLine(new Matrix(new[] { net.PartialDerivatives[1] }));
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Yellow;
                output = FeedForwardNeuralNetwork.ComputeLayerBatch(output, net.Layers[1].Item1, net.Layers[1].Item2);
                Console.WriteLine(output);
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.WriteLine(new Matrix(new[] { net.PartialDerivatives[2] }));
                Console.WriteLine();
                output = FeedForwardNeuralNetwork.ComputeLayerBatch(output, net.Layers[2].Item1, net.Layers[2].Item2);
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine(output.Stringify);

                Console.WriteLine();
                Console.WriteLine(output - new Matrix(new[] { new[] { target } }));
                //Console.WriteLine("     ");
            }

            Console.WriteLine();
            while (true)
            {
                Console.WriteLine(net.Compute(CHelper.RequestInput(@"Input the, well, inputs.", true, ConsoleColor.DarkYellow, ConsoleColor.Gray).Split(' ').ToFloats())[0]);
            }
        }

        private static async Task XORNetTest()
        {
            Random rand = new Random();
            var net = NeuralNetworkFactory.CreateRandomizedFeedForwardNeuralNetwork(rand, 2, (2, bin), (2, bin), (1, bin));
            await NeuralNetworkFactory.RandomTrain(net, rand, new[] { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f, } }, new[] { new[] { 0f }, new[] { 1f }, new[] { 1f }, new[] { 0f } });
            while (true)
            {
                Console.WriteLine(net.Compute(CHelper.RequestInput(@"Input the, well, inputs.", true, ConsoleColor.DarkYellow, ConsoleColor.Gray).Split(' ').ToFloats())[0]);
            }
        }

        private static void PerceptronTest()
        {
            Perceptron perceptron = new Perceptron(CHelper.RequestInput(@"Please input all weights, starting with the bias and separated by spaces.", true, ConsoleColor.DarkYellow, ConsoleColor.Gray).Split(' ').ToFloats());
            while (true)
            {
                Console.WriteLine(perceptron.Compute(CHelper.RequestInput(@"Input the, well, inputs.", true, ConsoleColor.DarkYellow, ConsoleColor.Gray).Split(' ').ToFloats()));
            }
        }

        private static async Task HillClimber()
        {
            Random random = new Random();
            Func<float[], float[], float>[] errorFuncs = new Func<float[], float[], float>[] { (a, b) => { float sumError = 0f; for (int i = 0; i < a.Length; i++) { sumError += Math.Abs(a[i] - b[i]); } return sumError / a.Length; }, (a, b) => { float sumError = 0f; for (int i = 0; i < a.Length; i++) { sumError += (a[i] - b[i]) * (a[i] - b[i]); } return sumError / a.Length; }, (a, b) => { float sumError = 0f; for (int i = 0; i < a.Length; i++) { sumError += (a[i] - b[i]) * (a[i] - b[i]); } return (float)Math.Sqrt(sumError / a.Length); } };
            ConsoleColor startingColor = Console.ForegroundColor;
            string input = CHelper.RequestInput("What is the target string?", true, ConsoleColor.DarkYellow, startingColor);
            Func<float[], float[], float> lossFunc = errorFuncs[CHelper.SelectorMenu("What is the error function?", new string[] { "\tMean Absolute Error", "\tMean Squared Error", "\tRoot Mean Squared Error" }, true, ConsoleColor.DarkYellow, startingColor, ConsoleColor.Magenta)];
            CHelper.CenteredWriteLine("Press any key to begin training", ConsoleColor.DarkYellow, Console.CursorTop + 1);
            Console.ReadKey(true);
            float[] characterVals = new float[input.Length];
            float[] newFandangledRandomString = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                newFandangledRandomString[i] = random.Next(1, 256);
                characterVals[i] = input[i];
            }

            bool done = false;
            int steps = 0;
            while (!done)
            {
                for (int i = 0; i < 50; i++)
                {
                    steps++;
                    if (UpdateStr() == 0)
                    {
                        done = true;
                        break;
                    }
                }
                Console.Clear();
                Console.ForegroundColor = startingColor;

                for (int i = 0; i < characterVals.Length; i++)
                {
                    Console.Write((char)newFandangledRandomString[i]);
                }

                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.Write(" => ");
                Console.ForegroundColor = startingColor;
                for (int i = 0; i < characterVals.Length; i++)
                {
                    Console.Write((char)characterVals[i]);
                }

                await Task.Delay(20);
            }
            Console.WriteLine();
            Console.WriteLine($"Done! Yay! Only took {steps} attempts!");
            Console.ReadKey();

            float UpdateStr()
            {
                float error = lossFunc(characterVals, newFandangledRandomString);
                int indexToChange = random.Next(0, characterVals.Length);
                int amountToAdd = random.Next(0, 2) * 2 - 1;
                newFandangledRandomString[indexToChange] += amountToAdd;
                float error2 = lossFunc(characterVals, newFandangledRandomString);

                if (error2 > error)
                {
                    newFandangledRandomString[indexToChange] -= amountToAdd;
                    return error;
                }
                return error2;
            }
        }


    }
    public abstract class CHelper
    {
        //Dictionary<char, string> dictionary
        //{
        //    get;
        //    set;
        //}
        public static async Task<int> SelectorMenu(string prompt, string[] options, bool wraps, ConsoleColor promptColor, ConsoleColor notSelectedColor, ConsoleColor selectedColor, CancellationToken cancelToken)
        {
            Console.ForegroundColor = promptColor;
            Console.WriteLine(prompt);
            int row = Console.CursorTop;
            int selectedIndex = 0;
            while (!cancelToken.IsCancellationRequested)
            {
                for (int i = 0; i < options.Length; i++)
                {
                    Console.ForegroundColor = i == selectedIndex ? selectedColor : notSelectedColor;
                    Console.SetCursorPosition(0, row + i);
                    Console.Write(options[i]);
                }

                ConsoleKeyInfo key = new ConsoleKeyInfo();
                while (!Console.KeyAvailable && !cancelToken.IsCancellationRequested)
                {
                }
                if (cancelToken.IsCancellationRequested)
                {
                    return -1;//await Task.FromCanceled<int>(cancelToken);
                }
                key = Console.ReadKey(true);
                if (key.Key == ConsoleKey.UpArrow)
                {
                    if (wraps)
                    {
                        selectedIndex = (selectedIndex - 1 + options.Length) % options.Length;
                    }
                    else
                    {
                        if (selectedIndex > 0)
                        {
                            selectedIndex--;
                        }
                    }
                }
                else if (key.Key == ConsoleKey.DownArrow)
                {
                    if (wraps)
                    {
                        selectedIndex = (selectedIndex + 1) % options.Length;
                    }
                    else
                    {
                        if (selectedIndex < options.Length - 1)
                        {
                            selectedIndex--;
                        }
                    }
                }
                else if (key.Key == ConsoleKey.Enter)
                {
                    return selectedIndex;
                }

            }

            return -1;
        }

        public static int SelectorMenu(string prompt, string[] options, bool wraps, ConsoleColor promptColor, ConsoleColor notSelectedColor, ConsoleColor selectedColor)
        {
            Console.ForegroundColor = promptColor;
            Console.WriteLine(prompt);
            int row = Console.CursorTop;
            int selectedIndex = 0;
            while (true)
            {
                for (int i = 0; i < options.Length; i++)
                {
                    Console.ForegroundColor = i == selectedIndex ? selectedColor : notSelectedColor;
                    Console.SetCursorPosition(0, row + i);
                    Console.Write(options[i]);
                }
                ConsoleKeyInfo key = Console.ReadKey(true);
                if (key.Key == ConsoleKey.UpArrow)
                {
                    if (wraps)
                    {
                        selectedIndex = (selectedIndex - 1 + options.Length) % options.Length;
                    }
                    else
                    {
                        if (selectedIndex > 0)
                        {
                            selectedIndex--;
                        }
                    }
                }
                else if (key.Key == ConsoleKey.DownArrow)
                {
                    if (wraps)
                    {
                        selectedIndex = (selectedIndex + 1) % options.Length;
                    }
                    else
                    {
                        if (selectedIndex < options.Length - 1)
                        {
                            selectedIndex--;
                        }
                    }
                }
                else if (key.Key == ConsoleKey.Enter)
                {
                    return selectedIndex;
                }

            }
        }

        public static async Task SlowWrite(ConsoleColor color, string text, int msDelay)
        {
            Console.ForegroundColor = color;
            var keyTask = Task.Run(() => Console.ReadKey(true));
            //keyTask.Start();
            for (int i = 0; i < text.Length; i++)
            {
                Console.Write(text[i]);
                var delayTask = Task.Delay(msDelay);
                //delayTask.Start();
                /*while (!delayTask.IsCompleted && !keyTask.IsCompleted)
                {
                    
                }
               

                if (keyTask.IsCompleted)*/
                if (Task.WaitAny(keyTask, delayTask) == 0)
                {
                    Console.Write(text.Substring(i + 1));
                    return;
                }
                //delayTask.Dispose();
                //keyTask.Dispose();
            }
        }

        public static async Task SlowWriteLine(ConsoleColor color, string text, int msDelay)
        {
            await SlowWrite(color, text + "\n", msDelay);
        }

        public static string RequestInput(string prompt, bool inputOnNewLine, ConsoleColor promptColor, ConsoleColor answerColor)
        {
            Console.ForegroundColor = promptColor;
            Console.Write(prompt + (inputOnNewLine ? '\n' : ' '));
            Console.ForegroundColor = answerColor;
            return Console.ReadLine();
        }
        public static string RequestInput(string prompt, bool inputOnNewLine, ConsoleColor promptColor, ConsoleColor answerColor, string[] autoCompleteOptions)
        {
            Console.ForegroundColor = promptColor;
            Console.Write(prompt + (inputOnNewLine ? '\n' : ' '));
            Console.ForegroundColor = answerColor;
            string str = "";
            while (true)
            {
                ConsoleKeyInfo key = Console.ReadKey(true);
                if (key.Key == ConsoleKey.Enter)
                {
                    Console.WriteLine();
                    return str;
                }
                if (key.Key == ConsoleKey.Backspace)
                {
                    if (str.Length > 0)
                    {
                        Console.Write("\x08");
                        Console.Write(" ");
                        Console.Write("\x08");
                        str = str.Remove(str.Length - 1, 1);
                    }
                }
                else if (key.Key == ConsoleKey.Tab)
                {
                    List<string> autoComplete = new List<string>();
                    for (int i = 0; i < autoCompleteOptions.Length; i++)
                    {
                        if (autoCompleteOptions[i].Contains(str))
                        {
                            autoComplete.Add(autoCompleteOptions[i]);
                        }
                    }
                    if (autoComplete.Count <= 0)
                    {
                        continue;
                    }
                    erase();
                    for (int i = 0; i < autoComplete[0].Length; i++)
                    {
                        bool x = false;
                        bool y = false;
                        foreach (string s in autoComplete)
                        {
                            if (i >= s.Length)
                            {
                                y = true;
                                break;
                            }
                            if (autoComplete[0][i] != s[i])
                            {
                                y = true;
                                x = true;
                                break;
                            }
                        }
                        if (y)
                        {
                            break;
                        }
                        if (x)
                        {
                            continue;
                        }
                        Console.Write(autoComplete[0][i]);
                        str += autoComplete[0][i];
                    }
                }
                else
                {
                    char c = key.KeyChar;
                    str += c;
                    Console.Write(c);

                }
                void erase()
                {
                    for (int i = 0; i < str.Length; i++)
                    {
                        Console.Write("\x08");
                        Console.Write(" ");
                        Console.Write("\x08");
                    }
                    str = "";
                }
            }
        }

        public static void CenteredWriteLine(string text, ConsoleColor color, int row)
        {
            Console.ForegroundColor = color;
            Console.SetCursorPosition(Console.BufferWidth / 2 - text.Length / 2, row);
            Console.WriteLine(text);
        }
        public static Dictionary<char, string> LoadASCIIFont(string asciiFontFile)
        {
            return JsonConvert.DeserializeObject<Dictionary<char, string>>(File.ReadAllText(asciiFontFile));
        }

        public static string ASCIIArt(string input, Dictionary<char, string> asciiFont)
        {
            //Dictionary<char, string> dictionary = JsonConvert.DeserializeObject<Dictionary<char, string>>(File.ReadAllText(asciiFontFile));
            StringBuilder builder = new StringBuilder();
            string[] vals = new string[asciiFont.Count];
            asciiFont.Values.CopyTo(vals, 0);
            for (int j = 0; j < vals[0].Split('\n').Length; j++)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    if (j < asciiFont[input[i]].Split('\n').Length)
                    {
                        builder.Append(asciiFont[input[i]].Split('\n')[j].Replace("\r", ""));
                    }
                }
                builder.AppendLine();

            }
            return builder.ToString();
        }
    }
}
