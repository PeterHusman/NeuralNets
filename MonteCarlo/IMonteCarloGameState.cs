using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.MonteCarlo
{
    public interface IMonteCarloGameState
    {
        bool IsMaxPlayerTurn { get; }
        bool IsTerminal { get; }
        double Value { get; }

        IEnumerable<IMonteCarloGameState> Moves { get; }
    }
}
