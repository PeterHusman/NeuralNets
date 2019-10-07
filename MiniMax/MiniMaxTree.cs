using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.MiniMax
{
    public static class MiniMaxTree
    {
        public static int MiniMax(IGameState state, bool isMax, int alpha = int.MinValue, int beta = int.MaxValue)
        {
            if(state.IsTerminal)
            {
                return state.Value;
            }

            if(isMax)
            {
                int val = int.MinValue;
                foreach(IGameState s in state.Moves)
                {
                    val = Math.Max(val, MiniMax(s, false, alpha, beta));
                    alpha = Math.Max(val, alpha);
                    if(alpha >= beta)
                    {
                       break;
                    }
                }
                return val;
            }
            else
            {
                int val = int.MaxValue;
                foreach (IGameState s in state.Moves)
                {
                    val = Math.Min(val, MiniMax(s, true, alpha, beta));
                    beta = Math.Min(val, beta);
                    if (alpha >= beta)
                    {
                        break;
                    }
                }
                return val;
            }
        }

        public static MiniMaxNode GenerateFromGameState(IGameState state)
        {
            return new MiniMaxNode(state);
        }
    }

    public class MiniMaxNode
    {
        public IGameState CurrentState { get; private set; }

        public int Score { get; private set; }

        public IReadOnlyList<MiniMaxNode> Children { get; private set; }

        public MiniMaxNode BestChild { get; private set; }

        private void PopulateChildren()
        {
            List<MiniMaxNode> nodes = new List<MiniMaxNode>();
            if (CurrentState.IsMaxPlayerTurn)
            {
                Score = int.MinValue;
                foreach (IGameState move in CurrentState.Moves)
                {
                    MiniMaxNode newNode = new MiniMaxNode(move);
                    nodes.Add(newNode);
                    if (newNode.Score > Score)
                    {
                        BestChild = newNode;
                        Score = BestChild.Score;
                    }
                }
            }
            else
            {
                Score = int.MaxValue;
                foreach (IGameState move in CurrentState.Moves)
                {
                    MiniMaxNode newNode = new MiniMaxNode(move);
                    nodes.Add(newNode);
                    if (newNode.Score < Score)
                    {
                        BestChild = newNode;
                        Score = BestChild.Score;
                    }
                }
            }
            Children = nodes.OrderByDescending(a => a.Score).ToArray();
        }

        internal MiniMaxNode(IGameState state)
        {
            CurrentState = state;
            if(state.IsTerminal)
            {
                Score = state.Value;
                Children = new MiniMaxNode[0];
                return;
            }
            PopulateChildren();
        }


    }
}
