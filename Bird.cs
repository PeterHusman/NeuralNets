using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnderEngine;

namespace NeuralNets
{
    public class Bird : BasePhysicsObject
    {
        public override ConsoleColor GetColor(Vector2 absolutePos)
        {
            return ConsoleColor.Red;
        }

        public Bird() : base()
        {
            HitBox = new HitBox(5, 5, 1, 1);
            Acceleration = new Vector2(0, 9.8f);
            Enabled = true;
        }
    }
}
