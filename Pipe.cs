using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnderEngine;

namespace NeuralNets
{
    public class Pipe : BasePhysicsObject
    {
        public override ConsoleColor GetColor(Vector2 absolutePos)
        {
            return ConsoleColor.Green;
        }

        public Pipe(int x, int y, int height) : base()
        {
            Velocity = new Vector2(-4, 0);
            HitBox = new HitBox(x, y, 1, height);
            Enabled = true;
        }
    }
}
