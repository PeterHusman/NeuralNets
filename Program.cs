using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NeuralNets.NeuralNetworks;
using System.Linq;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using UnderEngine;
using System.Diagnostics;
using NeuralNets.MiniMax;
using NeuralNets.MonteCarlo;
using NeuralNets.NeuralNetworks.Convolutional;
using System.Drawing;

namespace NeuralNets
{
    internal class Program
    {
        static Func<float, float> bin = a => a > 0 ? 1 : 0;
        private static async Task Main(string[] args)
        {
            //Modifying string!
            /*void LocalFunc()
            {
                ReadOnlySpan<char> stringSpan = "Hello World!".AsSpan();
                
                ref char c = ref MemoryMarshal.GetReference(stringSpan);
                for (int i = 0; i < 5; i++) {
                    ref char b = ref Unsafe.Add(ref c, i);
                    b = (char)((int)'a' + i);
                }
            }


            LocalFunc();*/



            int timeWasted = 0;
            string[] timeWastedStrings = { "Please make a selection.", "Really? I don't have all day.", "Seriously?!?", "I've had enough..." };
            while (true)
            {
                CancellationTokenSource source = new CancellationTokenSource();
                CancellationToken token = source.Token;
                var waitTask = Task.Delay(5000, token);
                int selection = -1;

                Task<int> chooseTask = Task.Run(() => CHelper.SelectorMenu(@"Please select the program to run.", new[] { "Hill Climber", "Perceptron", "XORNet - Random Train", "Gradient Descent", "Genetic Algorithm", "Game Trees", "Convolutional" },
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
                    case 4:
                        await GeneticAlgorithm();
                        break;
                    case 5:
                        await GameTrees();
                        break;
                    case 6:
                        await Convolutional();
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

        private static async Task Convolutional()
        {
            ConvolutionalNeuralNetwork convNet;
            float bestError;
            float[][][][] inputs;
            float[][][][] tOuts;
            switch (CHelper.SelectorMenu("Pick problem to solve", new[] { "MNIST", "Identify soul on field", "Random test", "Medium Test" }, true, ConsoleColor.Yellow, ConsoleColor.Gray, ConsoleColor.Magenta))
            {
                case 0:
                    throw new NotImplementedException("NMIST handwritten letters");
                    break;
                case 2:
                    convNet = new ConvolutionalNeuralNetwork(new ConvolutionalLayer(3, 2, 0, 1, 1, 1, ActivationFunctions.Identity), new PoolingLayer(2, 1, 0, 1, 1)/*, new PoolingLayer(2, 2, 0, 1, 1)*/);
                    convNet.Randomize(new Random());
                    bestError = float.PositiveInfinity;
                    inputs = new float[10][][][];
                    tOuts = new float[10][][][];
                    inputs[0] = new[] { new[] { new[] { 0f, 0f, 0f }, new[] { 0f, 0f, 0f }, new[] { 0f, 0f, 0f } } };
                    tOuts[0] = new[] { new[] { new[] { 0f, 0f }, new[] { 0f, 0f } } };
                    for (int i = 1; i < 10; i++)
                    {
                        inputs[i] = new[] { new[] { new[] { 0f, 0f, 0f }, new[] { 0f, 0f, 0f }, new[] { 0f, 0f, 0f } } };
                        inputs[i][0][(i - 1) / 3][(i - 1) % 3] = 1;
                        tOuts[i] = new float[1][][];
                        tOuts[i][0] = new float[2][];
                        for (int j = 0; j < tOuts[i][0].Length; j++)
                        {
                            tOuts[i][0][j] = new float[2];
                            for (int k = 0; k < tOuts[i][0][j].Length; k++)
                            {
                                tOuts[i][0][j][k] = (k <= ((i - 1) % 3) && ((i - 1) % 3) <= (k + 1) && j * 3 <= (i - 1) && (i - 1) < (j * 3 + 6)) ? 1f : 0f;
                            }
                        }
                    }
                    while (true)
                    {
                        float error = convNet.StochasticGradientDescent(inputs, tOuts, 0.01f);
                        if (error < bestError)
                        {
                            bestError = error;
                            Console.Clear();
                            Console.Write("Error: " + error);
                        }
                    }
                    break;
                case 1:
                    string[] files = Directory.GetFiles(@"C:\Users\Peter.Husman\Pictures\imgs");
                    bestError = float.PositiveInfinity;
                    inputs = new float[files.Length][][][];
                    tOuts = new float[files.Length][][][];
                    int wid = 0;
                    for (int i = 0; i < files.Length; i++)
                    {
                        inputs[i] = new float[1][][];
                        tOuts[i] = new[] { new[] { new[] { files[i].EndsWith("halfmat.jpg") ? 0f : 1f } } };
                        Bitmap bm = new Bitmap(files[i]);
                        wid = bm.Width;
                        for (int j = 0; j < inputs[i].Length; j++)
                        {
                            inputs[i][j] = new float[bm.Width][];
                            for (int k = 0; k < bm.Width; k++)
                            {
                                inputs[i][j][k] = new float[bm.Height];
                                for (int l = 0; l < bm.Height; l++)
                                {
                                    var pixel = bm.GetPixel(k, l);
                                    inputs[i][j][k][l] = j == 0 ? pixel.R : (j == 1 ? pixel.G : pixel.B);
                                }
                            }
                        }
                    }
                    var lyr = new ConvolutionalLayer(wid, 10, 0, 1, 1, 1, ActivationFunctions.Identity);
                    var lyr2 = new ConvolutionalLayer(lyr.OutputSideLength, lyr.OutputSideLength, 0, 1, 1, 1, ActivationFunctions.Identity);
                    convNet = new ConvolutionalNeuralNetwork(lyr, /*lyr2,*/ new PoolingLayer(lyr.OutputSideLength, lyr.OutputSideLength, 0, 1, 1), new ConvolutionalLayer(1, 1, 0, 1, 1, 1, ActivationFunctions.Identity)/*, new PoolingLayer(2, 2, 0, 1, 1)*/);
                    convNet.Randomize(new Random());
                    Console.Clear();
                    while (true)
                    {
                        float error = convNet.GradientDescent(inputs, tOuts, 0.00005e-9f); 
                        //if (error < bestError)
                        //{
                            bestError = error;
                            //Console.Clear();
                            Console.WriteLine("Error: " + error);
                        //}

                    }
                    break;
                case 3:
                    convNet = new ConvolutionalNeuralNetwork(new ConvolutionalLayer(3, 3, 0, 1, 1, 1, ActivationFunctions.ReLU)/*, new PoolingLayer(2, 2, 0, 1, 1)*/);
                    convNet.Randomize(new Random());
                    bestError = float.PositiveInfinity;
                    inputs = new[] {
                        new [] {new[] {
                            new [] { 0f, 0f, 0f },
                            new [] { 0f, 0f, 0f },
                            new [] { 0f, 0f, 0f } } },
                        new [] { new [] {
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 0f, 1f },
                            new [] { 1f, 1f, 1f } } },
                        new [] { new [] {
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 1f, 1f } } },
                        new [] { new [] {
                            new [] { 0f, 0f, 0f },
                            new [] { 0f, 1f, 0f },
                            new [] { 0f, 0f, 0f } } },
                        new [] { new [] {
                            new [] { 0f, 0f, 0f },
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 1f, 1f } } },
                        new [] { new [] {
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 1f, 1f },
                            new [] { 0f, 0f, 0f } } },
                        new [] { new [] {
                            new [] { 0f, 0f, 0f },
                            new [] { 1f, 1f, 1f },
                            new [] { 0f, 0f, 0f } } },
                        new [] { new [] {
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 0f, 1f },
                            new [] { 1f, 1f, 1f } } },
                        new [] { new [] {
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 0f, 1f },
                            new [] { 1f, 1f, 1f } } },
                        new [] { new [] {
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 0f, 1f },
                            new [] { 1f, 1f, 1f } } },
                        new [] { new [] {
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 0f, 1f },
                            new [] { 1f, 1f, 1f } } },
                        new [] { new [] {
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 0f, 1f },
                            new [] { 1f, 1f, 1f } } },
                        new [] { new [] {
                            new [] { 1f, 1f, 1f },
                            new [] { 1f, 0f, 1f },
                            new [] { 1f, 1f, 1f } } } };
                    tOuts = new[] {
                            new[] {
                                new [] {
                                    new[] {
                                        0f
                                    }
                                }
                        },
                        new [] {
                            new [] {
                                new [] {
                                    1f
                                }
                            }
                        }, new [] {
                            new [] {
                                new [] {
                                    0f
                                }
                            }
                        }, new [] {
                            new [] {
                                new [] {
                                    0f
                                }
                            }
                        }, new [] {
                            new [] {
                                new [] {
                                    0f
                                }
                            }
                        }, new [] {
                            new [] {
                                new [] {
                                    0f
                                }
                            }
                        }, new [] {
                            new [] {
                                new [] {
                                    0f
                                }
                            }
                        },
                        new [] {
                            new [] {
                                new [] {
                                    1f
                                }
                            }
                        },
                        new [] {
                            new [] {
                                new [] {
                                    1f
                                }
                            }
                        },
                        new [] {
                            new [] {
                                new [] {
                                    1f
                                }
                            }
                        },
                        new [] {
                            new [] {
                                new [] {
                                    1f
                                }
                            }
                        },
                        new [] {
                            new [] {
                                new [] {
                                    1f
                                }
                            }
                        },
                        new [] {
                            new [] {
                                new [] {
                                    1f
                                }
                            }
                        }
                    };
                    /*for (int i = 1; i < 10; i++)
                    {
                        inputs[i] = new[] { new[] { new[] { 0f, 0f, 0f }, new[] { 0f, 0f, 0f }, new[] { 0f, 0f, 0f } } };
                        inputs[i][0][(i - 1) / 3][(i - 1) % 3] = 1;
                        tOuts[i] = new float[1][][];
                        tOuts[i][0] = new float[2][];
                        for (int j = 0; j < tOuts[i][0].Length; j++)
                        {
                            tOuts[i][0][j] = new float[2];
                            for (int k = 0; k < tOuts[i][0][j].Length; k++)
                            {
                                tOuts[i][0][j][k] = (k <= ((i - 1) % 3) && ((i - 1) % 3) <= (k + 1) && j * 3 <= (i - 1) && (i - 1) < (j * 3 + 6)) ? 1f : 0f;
                            }
                        }
                    }*/
                    while (true)
                    {
                        float error = convNet.StochasticGradientDescent(inputs, tOuts, 0.01f);
                        if (error < bestError)
                        {
                            bestError = error;
                            Console.Clear();
                            Console.Write("Error: " + error);
                        }
                    }
                    break;
            }
        }

        private static async Task GameTrees()
        {
            if (CHelper.SelectorMenu("Select a type of game tree.", new string[] { "MiniMax", "MonteCarlo" }, true, ConsoleColor.Yellow, ConsoleColor.Gray, ConsoleColor.Magenta) == 0)
            {
                do
                {
                    var state = TicTacToeGameState.GenerateInitialState(3);
                    void DrawState()
                    {
                        Console.Clear();
                        for (int i = 0; i < 3; i++)
                        {
                            for (int j = 0; j < 3; j++)
                            {
                                Console.Write(state.Board[i][j] == TicTacToeSquareState.X ? "X " : (state.Board[i][j] == TicTacToeSquareState.O ? "O " : "  "));
                            }
                            Console.WriteLine();
                        }
                    }

                    while (!state.IsTerminal)
                    {
                        if (state.IsXTurn)
                        {
                            while (true)
                            {
                                var key = Console.ReadKey(true);
                                int n = key.Key - ConsoleKey.NumPad1;
                                if (n < 0 || n > 8)
                                {
                                    continue;
                                }
                                int a = 2 - (n / 3);
                                int b = n % 3;
                                if (state.Board[a][b] != TicTacToeSquareState.None)
                                {
                                    continue;
                                }
                                state.Board[a][b] = TicTacToeSquareState.X;
                                state.IsXTurn = false;
                                break;
                            }
                        }
                        else
                        {
                            TicTacToeGameState move = null;
                            int score = int.MaxValue;
                            foreach (var m in state.Moves)
                            {
                                int score2 = MiniMaxTree.MiniMax(m, ((TicTacToeGameState)m).IsXTurn);
                                if (score2 <= score)
                                {
                                    score = score2;
                                    if (score2 == -1)
                                    {
                                        move = (TicTacToeGameState)m;
                                        break;
                                    }
                                    move = (TicTacToeGameState)m;
                                }
                            }
                            if (move == null)
                            {
                                ;
                            }
                            state = move;

                        }
                        DrawState();
                    }

                    Console.WriteLine();
                    Console.WriteLine();
                    Console.WriteLine(state.Winning().ToString() + " wins!");
                } while (Console.ReadKey(true).Key == ConsoleKey.R);
            }
            else
            {
                do
                {
                    Random random = new Random();
                    MonteCarloNode<TicTacToeMonteCarloGameState> node = new MonteCarloNode<TicTacToeMonteCarloGameState>(TicTacToeMonteCarloGameState.GenerateInitialState(3), random);
                    void DrawState()
                    {
                        Console.Clear();
                        for (int i = 0; i < 3; i++)
                        {
                            for (int j = 0; j < 3; j++)
                            {
                                Console.Write(node.State.Board[i][j] == TicTacToeSquareState.X ? "X " : (node.State.Board[i][j] == TicTacToeSquareState.O ? "O " : "  "));
                            }
                            Console.WriteLine();
                        }
                    }

                    while (!node.IsTerminal)
                    {
                        if (node.State.IsXTurn)
                        {
                            while (true)
                            {
                                var key = Console.ReadKey(true);
                                int n = key.Key - ConsoleKey.NumPad1;
                                if (n < 0 || n > 8)
                                {
                                    continue;
                                }
                                int a = 2 - (n / 3);
                                int b = n % 3;
                                if (node.State.Board[a][b] != TicTacToeSquareState.None)
                                {
                                    continue;
                                }
                                node = node.Children.First(c => c.State.Board[a][b] == TicTacToeSquareState.X);
                                break;
                            }
                        }
                        else
                        {
                            node = node.TreeSearch(100);

                        }
                        DrawState();
                    }

                    Console.WriteLine();
                    Console.WriteLine();
                    Console.WriteLine(node.State.Winning().ToString() + " wins!");
                } while (Console.ReadKey(true).Key == ConsoleKey.R);
            }
        }

        private static async Task FlappyBird((FeedForwardNetwork net, double fitness)[] population, Random rand)
        {
            for (int i = 0; i < population.Length; i++)
            {
                population[i] = (new FeedForwardNetwork(3, (3, ActivationFunctions.BinaryStep), (4, ActivationFunctions.BinaryStep), (1, ActivationFunctions.BinaryStep)), 0);
                population[i].net.Randomize(rand);
            }

            TimeSpan fxd = TimeSpan.FromMilliseconds(16);
            TimeSpan time = fxd;
            int gen = 0;
            int seed = 0;

            int pipeDistance = 10;

            int score = 0;

            int bestScore = 0;

            int stepTime = 16;

            Stopwatch watch = new Stopwatch();

            while (true)
            {
                gen++;

                bool renderFrame = false;
                bool showAll = false;

                int onlyShow = 0;

                if (score > bestScore)
                {
                    bestScore = score;
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.BackgroundColor = ConsoleColor.Black;
                    Console.Clear();
                    Console.Write($"Gen:\t\t\t{gen}\nUpdates survived:\t{population[0].fitness}\nSim time survived:\t{TimeSpan.FromTicks((long)(population[0].fitness * time.Ticks))}\nScore:\t\t\t{score}");
                }

                if (Console.KeyAvailable)
                {
                    ConsoleKeyInfo key = Console.ReadKey(true);
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.BackgroundColor = ConsoleColor.Black;
                    Console.Clear();

                    if (key.Key == ConsoleKey.Spacebar)
                    {
                        Console.Write($"Gen:\t\t\t{gen}\nUpdates survived:\t{population[0].fitness}\nSim time survived:\t{TimeSpan.FromTicks((long)(population[0].fitness * time.Ticks))}\nScore:\t\t\t{score}");
                    }
                    else if (key.Key == ConsoleKey.V)
                    {
                        showAll = key.Modifiers.HasFlag(ConsoleModifiers.Shift);
                        if (!showAll)
                        {
                            for (int i = 0; i < population.Length; i++)
                            {
                                if (population[i].fitness >= population[onlyShow].fitness)
                                {
                                    onlyShow = i;
                                }
                            }
                        }
                        renderFrame = true;
                        gen--;
                        watch.Restart();
                        time = watch.Elapsed;
                    }
                }

                World world = new World(2, 30, 30, false);

                //Console.BufferWidth = Console.WindowWidth = 50;

                Pipe[] pipes = new Pipe[6];

                world.Objects = new ObjectBase[population.Length + pipes.Length];

                Bird[] birds = new Bird[population.Length];
                HashSet<Bird> flappingBirds = new HashSet<Bird>();
                Random random = new Random(seed);

                for (int i = 0; i < birds.Length; i++)
                {
                    birds[i] = new Bird();
                    world.Objects[i] = birds[i];
                    flappingBirds.Add(birds[i]);
                }



                for (int i = 0; i < pipes.Length; i++)
                {
                    int y = (i & 1) == 1 ? (int)(pipes[i - 1].HitBox.Size.Y + random.Next(4, 7)) : 0;
                    int height = y == 0 ? random.Next(4, 10) : (20 - y);
                    pipes[i] = new Pipe(12 + ((i / 2) * pipeDistance), y, height);
                    world.Objects[i + birds.Length] = pipes[i];
                }

                world.SetupObjects();

                Console.BackgroundColor = ConsoleColor.Black;

                score = 0;

                //Console.Clear();

                int pos = 0;

                while (flappingBirds.Count >= 1)
                {
                    //pos++;
                    /*for(int i = 0; i < world.Width; i++)
                    {
                        for(int j = 0; j < world.Height; j++)
                        {
                            world.ScreenBuffer[i, j] = ConsoleColor.Black;
                        }
                    }
                    Console.BackgroundColor = ConsoleColor.Black;
                    Console.Clear();*/

                    //world.Draw();
                    pos++;
                    if (renderFrame)
                    {
                        watch.Restart();
                        while (watch.ElapsedMilliseconds < stepTime) { }
                        time = fxd;//watch.Elapsed;
                        //watch.Restart();

                    }
                    world.Update(time);
                    if (renderFrame)
                    {
                        world.Draw();
                    }

                    for (int i = 0; i < birds.Length; i++)
                    {
                        if (!flappingBirds.Contains(birds[i]))
                        {
                            continue;
                        }
                        if (birds[i].Position.Y >= 20 || birds[i].Position.Y <= 1)
                        {
                            flappingBirds.Remove(birds[i]);
                            birds[i].Enabled = false;
                            if (renderFrame)
                            {
                                world.MarkDirty(birds[i].Position);
                            }
                            else
                            {
                                population[i].fitness = pos;
                            }
                            continue;
                        }
                        int nearestPipeX = 100;
                        int nearestGapY = 100;
                        bool valid = true;
                        for (int j = 0; j < pipes.Length; j++)
                        {
                            if (pipes[j].HitBox.Intersects(birds[i].HitBox))
                            {
                                flappingBirds.Remove(birds[i]);
                                birds[i].Enabled = false;
                                if (renderFrame)
                                {
                                    world.MarkDirty(birds[i].Position);
                                }
                                else
                                {
                                    population[i].fitness = pos;
                                }
                                valid = false;
                                break;
                            }

                            if (pipes[j].Position.X <= nearestPipeX + 1)
                            {
                                nearestPipeX = (int)pipes[j].Position.X;
                                nearestGapY = pipes[j].Position.Y == 0 ? nearestGapY : (int)pipes[j].Position.Y;
                            }
                        }

                        if ((!renderFrame || showAll || i == onlyShow) && (valid && population[i].net.Compute(new double[] { nearestPipeX - birds[i].Position.X, nearestGapY - birds[i].Position.Y, birds[i].Velocity.Y })[0] >= 1))
                        {
                            birds[i].Velocity.Y = -10;
                        }
                    }

                    int formerMostX = (int)pipes.Max(a => a.Position.X);

                    for (int i = 0; i < pipes.Length; i++)
                    {
                        if (pipes[i].Position.X <= 3)
                        {
                            int odd = i & 1;
                            score += odd;
                            int y = odd == 1 ? (int)(pipes[i - 1].HitBox.Size.Y + random.Next(4, 7)) : 0;
                            int height = y == 0 ? random.Next(4, 10) : (20 - y);
                            pipes[i] = new Pipe(formerMostX + pipeDistance, y, height);
                            world.Objects[i + birds.Length] = pipes[i];
                        }
                    }

                    if (Console.KeyAvailable && false)
                    {
                        var key = Console.ReadKey(true);
                        if (key.Key == ConsoleKey.PageUp)
                        {
                            stepTime++;
                        }
                        else if (key.Key == ConsoleKey.PageDown)
                        {
                            stepTime = (stepTime > 0) ? (stepTime - 1) : 0;
                        }
                    }



                    /*if(clear)
                    {
                        Console.ForegroundColor = ConsoleColor.Black;
                        Console.Clear();
                    }*/
                }

                if (!renderFrame)
                {
                    NeuralNetworkFactory.TrainGenetic(population, rand, 0.1f);
                }
                else
                {
                    time = fxd;
                }
            }
        }

        private static async Task GeneticAlgorithm()
        {
            int popSize = 100;

            Random rand = new Random();
            (FeedForwardNetwork net, double fitness)[] population;
            Console.Clear();

            int selectedProblem = CHelper.SelectorMenu("Please select the problem.", new[] { "Flappy Bird", "Dino Jump", "TicTacToe" }, true, ConsoleColor.DarkYellow, ConsoleColor.Gray, ConsoleColor.Magenta);

            population = new (FeedForwardNetwork, double)[popSize];

            switch (selectedProblem)
            {
                case 0:
                    await FlappyBird(population, rand);
                    break;
                case 1:
                    await DinoJump(population, rand);
                    break;
                case 2:
                    await GeneticTicTacToe(population, rand);
                    break;
            }



        }

        private static async Task GeneticTicTacToe((FeedForwardNetwork net, double fitness)[] population, Random rand)
        {
            for (int i = 0; i < population.Length; i++)
            {
                population[i] = (new FeedForwardNetwork(9, (18, ActivationFunctions.Identity), (1, ActivationFunctions.Identity)), 0);
                population[i].net.Randomize(rand);
            }

            double bestScore = double.NegativeInfinity;
            double otherBestScore = 0;

            int numberOfGames = 150;

            int gen = 0;

            MiniMaxNode node = MiniMaxTree.GenerateFromGameState(TicTacToeGameState.GenerateInitialState(3));

            while (true)
            {
                gen++;
                if (Console.KeyAvailable)
                {
                    var k = Console.ReadKey(true);
                    if (k.Key == ConsoleKey.Escape)
                    {
                        return;
                    }

                    do
                    {
                        var state = TicTacToeGameState.GenerateInitialState(3);
                        void DrawState()
                        {
                            Console.Clear();
                            for (int i = 0; i < 3; i++)
                            {
                                for (int j = 0; j < 3; j++)
                                {
                                    Console.Write(state.Board[i][j] == TicTacToeSquareState.X ? "X " : (state.Board[i][j] == TicTacToeSquareState.O ? "O " : "  "));
                                }
                                Console.WriteLine();
                            }
                        }

                        while (!state.IsTerminal)
                        {
                            if (state.IsXTurn)
                            {
                                int move = (int)population[0].net.Compute(state.Board.SelectMany(abc => abc.Select(bcd => (double)bcd)).ToArray())[0];
                                if (move < 0 || move > 8)
                                {
                                    break;
                                }
                                int a = move % 3;
                                int b = move / 3;
                                if (state.Board[a][b] != TicTacToeSquareState.None)
                                {
                                    break;
                                }
                                state.Board[a][b] = TicTacToeSquareState.X;

                                state.IsXTurn = !state.IsXTurn;
                            }
                            else
                            {
                                while (true)
                                {
                                    var key = Console.ReadKey(true);
                                    int n = key.Key - ConsoleKey.NumPad1;
                                    if (n < 0 || n > 8)
                                    {
                                        continue;
                                    }
                                    int a = 2 - (n / 3);
                                    int b = n % 3;
                                    if (state.Board[a][b] != TicTacToeSquareState.None)
                                    {
                                        continue;
                                    }
                                    state.Board[a][b] = TicTacToeSquareState.O;
                                    state.IsXTurn = !state.IsXTurn;
                                    break;
                                }

                            }
                            DrawState();
                        }

                        Console.WriteLine();
                        Console.WriteLine();
                        Console.WriteLine(state.Winning().ToString() + " wins!");
                    } while (Console.ReadKey(true).Key == ConsoleKey.R);
                }
                if (otherBestScore > bestScore)
                {
                    bestScore = otherBestScore;
                    Console.Clear();
                    Console.WriteLine("Generation: " + gen + "\nAvg. fitness: " + bestScore / numberOfGames);

                }
                for (int i = 0; i < population.Length; i++)
                {
                    double totalScore = 0;
                    for (int j = 0; j < numberOfGames; j++)
                    //Parallel.For(0, numberOfGames, j =>
                    {
                        MiniMaxNode currNode = node;

                        double roundScore = 0;

                        while (true)
                        {
                            TicTacToeGameState state = (TicTacToeGameState)currNode.CurrentState;
                            int move = (int)population[i].net.Compute(state.Board.SelectMany(abc => abc.Select(bcd => (double)bcd)).ToArray())[0];
                            if (move < 0 || move > 8)
                            {
                                roundScore -= 5;
                                break;
                            }
                            int a = move % 3;
                            int b = move / 3;
                            if (state.Board[a][b] != TicTacToeSquareState.None)
                            {
                                roundScore -= 5;
                                break;
                            }
                            currNode = currNode.Children.First(c => ((TicTacToeGameState)c.CurrentState).Board[a][b] == TicTacToeSquareState.X);
                            state = (TicTacToeGameState)currNode.CurrentState;

                            if (state.IsTerminal)
                            {

                                //roundScore = state.Winning() == TicTacToeSquareState.None ? 1 : 0;
                                break;
                            }

                            roundScore++;

                            int n = currNode.Children.Count;// * currNode.Children.Count;

                            currNode = currNode.Children[rand.Next(0, n)];
                            state = (TicTacToeGameState)currNode.CurrentState;

                            if (state.IsTerminal)
                            {
                                //roundScore = state.Winning() == TicTacToeSquareState.None ? 1 : 0;
                                break;
                            }

                        }

                        if (currNode.CurrentState.IsTerminal)
                        {
                            TicTacToeSquareState winningParty = ((TicTacToeGameState)currNode.CurrentState).Winning();
                            if (winningParty == TicTacToeSquareState.X)
                            {
                                roundScore = 40 - roundScore;
                            }
                            else if (winningParty == TicTacToeSquareState.O)
                            {
                                roundScore -= 5;
                            }
                            else
                            {

                            }
                        }

                        totalScore += roundScore;
                    }
                    //);
                    if (totalScore > otherBestScore)
                    {
                        otherBestScore = totalScore;
                    }
                }

                NeuralNetworkFactory.TrainGenetic(population, rand, 0.05f);
            }
        }

        private static async Task DinoJump((FeedForwardNetwork net, double fitness)[] population, Random rand)
        {
            for (int i = 0; i < population.Length; i++)
            {
                population[i] = (new FeedForwardNetwork(3, (3, ActivationFunctions.BinaryStep), (4, ActivationFunctions.BinaryStep), (2, ActivationFunctions.BinaryStep)), 0);
                population[i].net.Randomize(rand);
            }

            int bestScore = -1;
            int gen = 0;
            int pos = 0;
            int score = 0;

            TimeSpan elapsedTime = TimeSpan.FromMilliseconds(16);
            int millis = 16;

            Stopwatch watch = new Stopwatch();
            Random random = new Random(0);
            Func<int> getY = () => random.Next(13, 15);

            ConsoleKeyInfo keyHit = new ConsoleKeyInfo();
            ConsoleKey defaultKey = keyHit.Key;

            while (true)
            {
                random = new Random(0);
                Dinosaur[] dinosaurs = new Dinosaur[population.Length];

                Obstacle[] obstacles = new Obstacle[10];

                int highestX = 1;

                for (int i = 0; i < obstacles.Length; i++)
                {
                    obstacles[i] = new Obstacle(highestX = highestX + 3, getY(), random.Next(1, 3), 1, -5);
                }

                for (int i = 0; i < dinosaurs.Length; i++)
                {
                    dinosaurs[i] = new Dinosaur(obstacles);
                }

                World world = new World(10, 30, 17, false);
                world.Objects = dinosaurs.Concat<BasePhysicsObject>(obstacles).ToArray();
                world.SetupObjects();
                gen++;

                bool cont = false;
                bool renderFrame = false;

                if (score > bestScore)
                {
                    bestScore = score;
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.BackgroundColor = ConsoleColor.Black;
                    Console.Clear();
                    Console.Write($"Gen:\t\t\t{gen}\nUpdates survived:\t{population[0].fitness}\nSim time survived:\t{TimeSpan.FromTicks((long)(population[0].fitness * elapsedTime.Ticks))}\nScore:\t\t\t{bestScore}");
                }

                if (keyHit.Key != defaultKey)
                {
                    ConsoleKeyInfo key = keyHit;
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.BackgroundColor = ConsoleColor.Black;
                    Console.Clear();

                    if (key.Key == ConsoleKey.Spacebar)
                    {
                        Console.Write($"Gen:\t\t\t{gen}\nUpdates survived:\t{population[0].fitness}\nSim time survived:\t{TimeSpan.FromTicks((long)(population[0].fitness * elapsedTime.Ticks))}\nScore:\t\t\t{bestScore}");
                    }
                    else if (key.Key == ConsoleKey.V)
                    {
                        /*showAll = key.Modifiers.HasFlag(ConsoleModifiers.Shift);
                        if (!showAll)
                        {
                            for (int i = 0; i < population.Length; i++)
                            {
                                if (population[i].fitness >= population[onlyShow].fitness)
                                {
                                    onlyShow = i;
                                }
                            }
                        }*/
                        renderFrame = true;
                        //gen--;
                        watch.Restart();
                    }

                    keyHit = new ConsoleKeyInfo();
                }
                pos = 0;
                score = 0;

                do
                {
                    if (Console.KeyAvailable)
                    {
                        var key = Console.ReadKey(true);
                        if (key.Key == ConsoleKey.V && key.Modifiers.HasFlag(ConsoleModifiers.Shift))
                        {
                            renderFrame = !renderFrame;
                        }
                        else if (key.Key == ConsoleKey.U)
                        {
                            Console.ForegroundColor = ConsoleColor.White;
                            Console.BackgroundColor = ConsoleColor.Black;
                            Console.Clear();
                            Console.Write($"Gen in progress:\t{gen}\nUpdates survived:\t{pos}\nSim time survived:\t{TimeSpan.FromTicks((long)(pos * elapsedTime.Ticks))}\nScore:\t\t\t{score}");
                        }
                        else if (key.Key == ConsoleKey.Escape)
                        {
                            break;
                        }
                        else if (key.Key == ConsoleKey.PageUp)
                        {
                            millis = Math.Max(millis - 5, 0);
                        }
                        else if (key.Key == ConsoleKey.PageDown)
                        {
                            millis += 5;
                        }
                        else
                        {
                            keyHit = key;
                        }
                    }

                    cont = false;
                    pos++;
                    if (renderFrame)
                    {
                        watch.Restart();
                        while (watch.ElapsedMilliseconds < millis) { }
                    }
                    world.Update(elapsedTime);
                    if (renderFrame)
                    {
                        world.Draw();
                    }
                    Obstacle nearest = null;
                    foreach (Obstacle item in obstacles)
                    {
                        if (item.Position.X > 2 && item.Position.X <= (nearest?.Position.X ?? 150))
                        {
                            nearest = item;
                        }
                    }
                    for (int i = 0; i < dinosaurs.Length; i++)
                    {
                        if (!dinosaurs[i].Enabled)
                        {
                            continue;
                        }
                        bool alive = dinosaurs[i].TryKill();
                        cont = cont || alive;
                        if (alive)
                        {
                            dinosaurs[i].ProcessNetOutputs(population[i].net.Compute(new double[] { nearest.Position.X - dinosaurs[i].Position.X, nearest.Position.Y - dinosaurs[i].Position.Y, dinosaurs[i].Velocity.Y }));
                            continue;
                        }
                        population[i].fitness = pos;
                    }
                    for (int i = 0; i < obstacles.Length; i++)
                    {
                        if (obstacles[i].Position.X <= 1)
                        {
                            obstacles[i].Reset(highestX, getY());
                            score++;
                        }
                    }
                } while (cont);
                //if (!renderFrame)
                //{
                NeuralNetworkFactory.TrainGenetic(population, rand, 0.1f);
                //}
            }
        }

        private static async Task GradientTest()
        {
            Random rand = new Random();
            FeedForwardNetwork net = null;
            double[][] inputs = new double[0][];
            double[][] outputs = new double[0][];
            Console.Clear();
            int selectedProblem = CHelper.SelectorMenu("Please select the problem.", new[] { "XOR", "Sine" }, true, ConsoleColor.DarkYellow, ConsoleColor.Gray, ConsoleColor.Magenta);
            switch (selectedProblem)
            {
                case 0:
                    net = new FeedForwardNetwork(2, (2, ActivationFunctions.Sigmoid), (4, ActivationFunctions.Sigmoid), (1, ActivationFunctions.Sigmoid));
                    inputs = new double[][] { new double[] { 0, 0 }, new double[] { 0, 1 }, new double[] { 1, 0, }, new double[] { 1, 1 } };
                    outputs = new double[][] { new double[] { 0 }, new double[] { 1 }, new double[] { 1f }, new double[] { 0f } };
                    break;
                case 1:
                    net = new FeedForwardNetwork(1, (5, ActivationFunctions.TanH), (1, ActivationFunctions.TanH));
                    inputs = new double[100][];
                    outputs = new double[100][];
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        inputs[i] = new[] { Math.PI * 2 * i / inputs.Length };
                        outputs[i] = new[] { Math.Sin(inputs[i][0]) };
                    }
                    break;
            }

            net.Randomize(rand);

            double error = 1;

            double threshold = 0.0001;

            double[][] realOuts = outputs;

            //Thread train = new Thread(Descent);

            //BackgroundWorker backgroundWorker = new BackgroundWorker();
            //backgroundWorker.WorkerSupportsCancellation = true;
            //backgroundWorker.DoWork += Descend;
            //backgroundWorker.RunWorkerCompleted += (a, b) => { lock (locker) { error = (double)b.Result; realOuts = dubs; } if (!b.Cancelled) { backgroundWorker.RunWorkerAsync(); } };
            //backgroundWorker.RunWorkerAsync();

            //train.Start();

            Console.Clear();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Training...\nMSE: ");

            bool cond1 = true;
            int a = 0;

            while (cond1 && !Console.KeyAvailable)
            {
                error = net.GradientDescent(inputs, outputs, 0.01, 0.2, out realOuts);

                if (a < 500)
                {
                    a++;
                    continue;
                }

                a = 0;

                Console.SetCursorPosition(5, 1);
                Console.ForegroundColor = ConsoleColor.Yellow;

                cond1 = error >= threshold;

                Console.Write(error);

                if (selectedProblem == 1)
                {

                    Console.WriteLine();
                    Console.WriteLine();

                    double[] vals = new double[100];
                    double[] targetOuts = new double[vals.Length];
                    for (int i = 0; i < vals.Length; i++)
                    {
                        int scaledInd = i * 1;
                        vals[i] = realOuts[scaledInd][0];
                        targetOuts[i] = outputs[scaledInd][0];
                    }

                    int verticalSubdivisions = 30;
                    double step = 2f / verticalSubdivisions;

                    for (int i = verticalSubdivisions; i >= 0; i--)
                    {
                        for (int j = 0; j < vals.Length; j++)
                        {
                            double lower = step * i - 1;
                            double upper = lower + step;
                            if (vals[j] >= lower && vals[j] <= upper)
                            {
                                Console.BackgroundColor = ConsoleColor.Yellow;
                            }
                            else if (targetOuts[j] >= lower && targetOuts[j] <= upper)
                            {
                                Console.BackgroundColor = ConsoleColor.Red;
                            }
                            else
                            {
                                Console.BackgroundColor = ConsoleColor.Black;
                            }
                            Console.Write(' ');
                        }
                        Console.WriteLine();
                    }
                }
            }

            //backgroundWorker.CancelAsync();

            //train.Join();

            Console.WriteLine();
            while (true)
            {
                Console.WriteLine(net.Compute(CHelper.RequestInput(@"Input the, well, inputs.", true, ConsoleColor.DarkYellow, ConsoleColor.Gray).Split(' ').Select(abc => double.Parse(abc)).ToArray())[0]);
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
