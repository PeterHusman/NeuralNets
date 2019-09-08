using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.MiniMax
{
    public class MiniMaxTree
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
    }
}
