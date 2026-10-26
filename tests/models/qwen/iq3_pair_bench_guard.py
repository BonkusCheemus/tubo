#!/usr/bin/env python3
"""CPU execution of actual host parity-gate/argument checks with HIP memory stubs.

Not GPU execution: the source launch callback is replaced with full/missing
stores. Removing the actual poison must make the missing-store regression fail.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]


def block(source, start):
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


SHIM = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
template<class T> struct Buffer {
  std::vector<T> data;
  explicit Buffer(std::size_t n):data(n){}
  T* Get(){return data.data();}
};
struct Geometry {const char* name="pair";std::function<void(const void*)> pair_reference,launch;};
constexpr int hipMemcpyDeviceToHost=0;
void Check(int code){assert(code==0);}
int hipGetLastError(){return 0;}
int hipMemset(void* p,int v,std::size_t n){std::memset(p,v,n);return 0;}
int hipMemcpy(void* dst,const void* src,std::size_t n,int){std::memcpy(dst,src,n);return 0;}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    if os.getloadavg()[0] > float(os.environ.get('IQ3_PAIR_MAX_LOAD', '3')):
        raise SystemExit('Paused: load exceeds IQ3_PAIR_MAX_LOAD')
    bench = (ROOT/'tools/qwen27b/dflash_gemm_bench.hip').read_text()
    start = bench.index('std::vector<float> pair_expected;')
    launch = bench[start:bench.index('Compare<<<', start)]
    compare = block(bench, bench.index('if (!pair_expected.empty())', start))
    header = (ROOT/'src/models/qwen/hip/kernels/small_batch_gemm.hpp').read_text()
    fits = block(header, header.index('constexpr bool FitsSmallBatch32BitIndices'))
    start = bench.index('const std::size_t column_block')
    validation = bench[start:bench.index('const std::size_t copies', start)]
    prefix = SHIM + fits + r'''
template<std::size_t Batch> void accept(const std::string& type,std::size_t rows,
  std::size_t columns,int iterations,bool fused=false,const std::string& up_type="") {
''' + validation + r'''
}
int main() {
  accept<8>("iq3-s",5121,5120,24);
  accept<8>("iq3-xxs",5121,5120,24);
  for(int missing: {0,1,40}) {
    const std::size_t batch=8,rows=5,n=batch*rows;
    Buffer<float> output(n);Buffer<unsigned> mismatch(1);Buffer<std::uint8_t> mapped_weights(1);
    Geometry geometry;
    geometry.pair_reference=[&](const void*){for(std::size_t i=0;i<n;++i)output.Get()[i]=float(i+1)*0.125F;};
    geometry.launch=[&](const void*){for(std::size_t i=0;i<n-missing;++i)output.Get()[i]=float(i+1)*0.125F;};
    bool rejected=false;
    try {
'''
    suffix = r'''
    } catch(const std::runtime_error&) {rejected=true;}
    if(rejected!=(missing!=0)) return 2;
  }
}
'''
    mutant, count = re.subn(r'Check\(hipMemset\(output\.Get\(\), 0xFF,\s*batch \* rows \* sizeof\(float\)\)\);', ';', launch)
    assert count == 1, 'Cannot identify actual poison statement'
    receipt = {}
    with tempfile.TemporaryDirectory() as directory:
        cpp, binary = Path(directory)/'guard.cpp', Path(directory)/'guard'
        for name, code in [('actual', launch), ('poison_removed', mutant)]:
            cpp.write_text(prefix + code + compare + suffix)
            subprocess.run([args.compiler,'-std=c++20','-O2',str(cpp),'-o',str(binary)],check=True)
            result = subprocess.run([str(binary)])
            assert (result.returncode == 0) == (name == 'actual')
            receipt[name] = result.returncode
    receipt['checked'] = 'Actual host gate accepts complete stores; rejects one missing tail/all missing stores; actual argument/index checks accept both IQ3 5121x5120 B8 cases.'
    receipt['limit'] = 'HIP memory APIs and launch callbacks are CPU stubs; native GPU parity/tails remain unrun.'
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
