using NeuralNets.MiniMax;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.MonteCarlo
{
    public class MonteCarloNode<T> where T : IGameState
    {
        public T State { get; private set; }

        public IEnumerable<MonteCarloNode<T>> Children
        {
            get
            {
                if (children == null)
                {
                    children = State.Moves.Select(a => new MonteCarloNode<T>((T)a, this, Random));
                }
                return children;
            }
            private set => children = value;
        }

        public double CumulativeResult { get; private set; } = 0;

        public int SimulationCount { get; private set; } = 0;

        public bool Visited { get; private set; } = false;

        public bool FullyExpanded { get; private set; } = false;

        public MonteCarloNode<T> Parent { get; private set; } = null;

        public Random Random { get; private set; }

        public MonteCarloNode<T> TreeSearch()
        {
            Visited = true;
            for (int i = 0; i < 1000; i++)
            {
                MonteCarloNode<T> currLeaf = Traverse(this);
                double result = Rollout(currLeaf);
                BackPropogate(currLeaf, result);
                if(!FullyExpanded && Children.All(a => a.Visited))
                {
                    FullyExpanded = true;
                }
            }
            return OptimalChild();
        }

        public MonteCarloNode(T state, MonteCarloNode<T> parent, Random random)
        {
            State = state;
            Parent = parent;
            Random = random;
        }

        public MonteCarloNode(T state, Random random)
        {
            State = state;
            Random = random;
        }

        public MonteCarloNode<T> OptimalChild()
        {
            return Children.OrderByDescending(a => a.SimulationCount).FirstOrDefault(/*a => a.CumulativeResult / a.SimulationCount > 0.9*/);
        }

        public static void BackPropogate(MonteCarloNode<T> node, double result)
        {
            while (node != null)
            {
                //Figure out use of result to allow for computation relative to player
                node.CumulativeResult += result;
                node.SimulationCount++;
                node = node.Parent;
                //Try this:
                //result = 1 - result;
            }
        }

        public static double Rollout(MonteCarloNode<T> n)
        {
            while (!n.IsTerminal)
            {
                n = n.RolloutPolicy();
            }
            return n.State.Value;
        }

        public MonteCarloNode<T> RolloutPolicy()
        {
            MonteCarloNode<T>[] nodes = Children.ToArray();
            return nodes[Random.Next(0, nodes.Length)];
        }

        const double RootTwo = 1.41421356;
        private IEnumerable<MonteCarloNode<T>> children = null;

        public double UCT()
        {
            return (CumulativeResult / SimulationCount) + RootTwo * Math.Sqrt(Math.Log10(Parent.SimulationCount) / SimulationCount);
        }

        private MonteCarloNode<T> BestUCT()
        {
            double bestUCT = double.NegativeInfinity;
            MonteCarloNode<T> correspondingNode = null;
            foreach (MonteCarloNode<T> node in Children)
            {
                double d = node.UCT();
                if (d > bestUCT)
                {
                    bestUCT = d;
                    correspondingNode = node;
                }
            }
            return correspondingNode;
        }

        public bool IsTerminal => State.IsTerminal;

        public MonteCarloNode<T> PickUnvisited()
        {
            /*var a = Children().Where(b => !b.Visited).ToArray();
            return a[random.Next(0, a.Length)];*/
            return Children.First(a => !a.Visited);
        }

        public static MonteCarloNode<T> Traverse(MonteCarloNode<T> n)
        {
            while (n.FullyExpanded)
            {
                n = n.BestUCT();
            }
            return n.IsTerminal ? n : n.PickUnvisited();
        }
    }
}
