using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    public ref struct FileReader
    {
        private ReadOnlySpan<byte> file;

        public ReadOnlySpan<byte> BackingSpan { get => file; }

        public FileReader(ReadOnlySpan<byte> file)
        {
            this.file = file;
        }


        public uint ReadU4()
        {
            uint toRet = (uint)(file[0] << 24) | (uint)(file[1] << 16) | (uint)(file[2] << 8) | (uint)file[3];
            file = file.Slice(4);
            return toRet;
        }

        public FileReader ReadLength(int length)
        {
            var toRet = file.Slice(0, length);
            file = file.Slice(length);
            return new FileReader(toRet);
        }

        public FileReader ReadLength(uint length)
        {
            return ReadLength((int)length);
        }

        public uint ReadU2()
        {
            return (ReadU1() << 8) | ReadU1();
        }

        public uint ReadU1()
        {
            uint toRet = file[0];
            file = file.Slice(1);
            return toRet;
        }
    }
}
