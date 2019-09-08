using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.MiniMax
{
    public enum TicTacToeSquareState
    {
        None = 0,
        O = 2,
        X = 1
    }

    public class TicTacToeGameState : IGameState
    {
        public bool IsTerminal => Winning() != TicTacToeSquareState.None || !Board.Any(a => a.Any(b => b == TicTacToeSquareState.None));

        public TicTacToeSquareState Winning()
        {
            for (int i = 0; i < Board.Length; i++)
            {
                TicTacToeSquareState curr = Board[i][0];
                bool valid = true;
                for (int j = 1; j < Board.Length; j++)
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
                for (int j = 1; j < Board.Length; j++)
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
            for (int i = 1; i < Board.Length; i++)
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
            for (int i = 1; i < Board.Length; i++)
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

        public int Value
        {
            get
            {
                var a = Winning();
                return a == TicTacToeSquareState.X ? 1 : (a == TicTacToeSquareState.O ? -1 : 0);
            }
        }

        public TicTacToeSquareState[][] Board { get; private set; }

        public bool IsXTurn { get; set; }

        public IEnumerable<IGameState> Moves
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

                        yield return new TicTacToeGameState(newBoard, false, !IsXTurn);
                    }
                }
            }
        }

        public static TicTacToeGameState GenerateInitialState(int sideLength) => new TicTacToeGameState(new TicTacToeSquareState[sideLength][], true, true);

        private TicTacToeGameState(TicTacToeSquareState[][] board, bool initState, bool isXTurn)
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
