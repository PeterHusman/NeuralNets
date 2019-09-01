using NeuralNets.NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnderEngine;

namespace NeuralNets
{
    public class Dinosaur : BasePhysicsObject
    {
        public override ConsoleColor GetColor(Vector2 absolutePos)
        {
            return ConsoleColor.Red;
        }

        public Dinosaur(Obstacle[] obstacles)
        {
            Obstacles = obstacles;
            Position = new Vector2(2, 13);
            HitBox.Size = new Vector2(1, 2);
            Enabled = true;
        }

        public Obstacle[] Obstacles { get; set; }

        private bool ducking = false;

        public bool Ducking
        {
            set
            {
                HitBox.Size.Y = value ? 1 : 2;
                if (value && !ducking)
                {
                    Position.Y++;
                    Velocity.Y = 10;
                }
                else if(ducking && !value)
                {
                    Position.Y--;
                }
                ducking = value;
            }
        }

        public bool KillCheck()
        {
            //TODO: Benchmark and determine which method is fastest.

            //bool ret = false;
            //Parallel.ForEach(Obstacles, a => { ret = ret || a.HitBox.Intersects(HitBox); });
            //return ret;

            //return Obstacles.Any(a => a.HitBox.Intersects(HitBox));

            for (int i = 0; i < Obstacles.Length; i++)
            {
                if (Obstacles[i].HitBox.Intersects(HitBox))
                {
                    return true;
                }
            }
            return false;
        }

        public void Kill()
        {
            HitBox.Enabled = false;
        }

        public bool TryKill()
        {
            return HitBox.Enabled = !KillCheck();
        }

        public override void Update(TimeSpan elapsedTime)
        {
            base.Update(elapsedTime);

            if(Position.Y + HitBox.Size.Y >= 15)
            {
                Position.Y = 15 - HitBox.Size.Y;
                Velocity.Y = Math.Min(Velocity.Y, 0);
            }
        }

        public void ProcessNetOutputs(double[] netOutputs)
        {
            if(Position.Y >= 15 && netOutputs[0] >= 1)
            {
                Velocity.Y = -5;
            }
            Ducking = netOutputs[1] >= 1;
        }
    }

    public class Obstacle : BasePhysicsObject
    {
        public override ConsoleColor GetColor(Vector2 absolutePos)
        {
            return ConsoleColor.White;
        }

        public Obstacle(Obstacle[] allObstacles, int space, int y, int width, int height, int speed)
        {
            float x = allObstacles.Max(a => a?.Position.X ?? 0);

            Position = new Vector2(x + space, y);
            HitBox.Size = new Vector2(width, height);

            Velocity = new Vector2(speed, 0);
            Enabled = true;
        }

        public Obstacle(int x, int y, int width, int height, int speed)
        {
            Position = new Vector2(x, y);
            HitBox.Size = new Vector2(width, height);

            Velocity = new Vector2(speed, 0);
        }
    }
}
