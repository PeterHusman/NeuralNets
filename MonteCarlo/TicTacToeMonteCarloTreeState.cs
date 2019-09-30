using NeuralNets.MiniMax;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.MonteCarlo
{
    public class TicTacToeMonteCarloGameState : IMonteCarloGameState
    {
        bool IMonteCarloGameState.IsMaxPlayerTurn => IsXTurn;

        public bool IsTerminal => Winning() != TicTacToeSquareState.None || !Board.Any(a => a.Any(b => b == TicTacToeSquareState.None));

        public TicTacToeSquareState Winning()
        {
            for (int i = 0; i < Board.Length; i++)
            {
                TicTacToeSquareState curr = Board[i][0];
                bool valid = true;
                for (int j = 0; j < Board.Length; j++)
                {
                    if (curr != Board[i][j])
                    {
                        valid = false;
                        break;
                    }
                }
                if (valid && curr != TicTacToeSquareState.None)
                {
                    return curr;
                }
            }

            for (int i = 0; i < Board.Length; i++)
            {
                TicTacToeSquareState curr = Board[0][i];
                bool valid = true;
                for (int j = 0; j < Board.Length; j++)
                {
                    if (curr != Board[j][i])
                    {
                        valid = false;
                        break;
                    }
                }
                if (valid && curr != TicTacToeSquareState.None)
                {
                    return curr;
                }
            }

            TicTacToeSquareState curr2 = Board[0][0];
            bool valid2 = true;
            for (int i = 0; i < Board.Length; i++)
            {
                if (curr2 != Board[i][i])
                {
                    valid2 = false;
                    break;
                }
            }
            if (valid2 && curr2 != TicTacToeSquareState.None)
            {
                return curr2;
            }

            valid2 = true;
            curr2 = Board[Board.Length - 1][0];
            for (int i = 0; i < Board.Length; i++)
            {
                if (curr2 != Board[Board.Length - 1 - i][i])
                {
                    valid2 = false;
                    break;
                }
            }
            if (valid2 && curr2 != TicTacToeSquareState.None)
            {
                return curr2;
            }

            return TicTacToeSquareState.None;
        }

        public double Value
        {
            get
            {
                TicTacToeSquareState winning = Winning();
                if(winning == TicTacToeSquareState.None)
                {
                    return 0.5;
                }

                return winning == TicTacToeSquareState.X ? 0 : 1;
            }
        }

        public TicTacToeSquareState[][] Board { get; private set; }

        public bool IsXTurn { get; set; }

        public IEnumerable<IMonteCarloGameState> Moves
        {
            get
            {
                TicTacToeSquareState next = IsXTurn ? TicTacToeSquareState.X : TicTacToeSquareState.O;
                for (int i = 0; i < Board.Length; i++)
                {
                    for (int j = 0; j < Board.Length; j++)
                    {
                        if (Board[i][j] != TicTacToeSquareState.None)
                        {
                            continue;
                        }
                        TicTacToeSquareState[][] newBoard = new TicTacToeSquareState[Board.Length][];
                        for (int k = 0; k < newBoard.Length; k++)
                        {
                            newBoard[k] = Board[k].ToArray();
                            if (k == i)
                            {
                                newBoard[k][j] = next;
                            }
                        }

                        yield return new TicTacToeMonteCarloGameState(newBoard, false, !IsXTurn);
                    }
                }
            }
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    s += Board[i][j] == TicTacToeSquareState.X ? "X " : (Board[i][j] == TicTacToeSquareState.O ? "O " : "  ");
                }
                s += "\\\\\n";
            }
            //s += MiniMaxTree.MiniMax(this, IsXTurn);
            return s;
        }

        public static TicTacToeMonteCarloGameState GenerateInitialState(int sideLength) => new TicTacToeMonteCarloGameState(new TicTacToeSquareState[sideLength][], true, true);

        private TicTacToeMonteCarloGameState(TicTacToeSquareState[][] board, bool initState, bool isXTurn)
        {
            IsXTurn = isXTurn;
            Board = board;
            int numOfEmpties = 0;
            if (initState)
            {
                for (int i = 0; i < Board.Length; i++)
                {
                    Board[i] = new TicTacToeSquareState[Board.Length];
                }
                numOfEmpties = 9;
            }
            else
            {
                for (int i = 0; i < Board.Length; i++)
                {
                    for (int j = 0; j < Board.Length; j++)
                    {
                        if (Board[i][j] == TicTacToeSquareState.None)
                        {
                            numOfEmpties++;
                        }
                    }
                }
            }


        }
    }
}
