using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.MiniMax
{
    public interface IGameState
    {
        bool IsMaxPlayerTurn { get; }

        bool IsTerminal { get; }
        int Value { get; }

        IEnumerable<IGameState> Moves { get; }
    }
}
