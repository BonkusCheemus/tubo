#!/usr/bin/env python3
"""Execute the actual IQ3 R1/R2/R4 helper bodies on CPU, not HIP/codegen tests.

The host shim implements half conversion, signed dot4 and sign-byte application.
Noncontracting mode also checks an independent GGUF byte-index oracle. FMA
mode compares actual R2/R4 against R1 only and requires emitted FMA
instructions. Synthetic grid entries distinguish both IQ3_S index bits; no
model/tensor payload is loaded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / 'src/models/qwen/hip/kernels/small_batch_dot4.hpp'


def guard():
    if os.getloadavg()[0] > float(os.environ.get('IQ3_PAIR_MAX_LOAD', '3')):
        raise SystemExit('Paused: load exceeds IQ3_PAIR_MAX_LOAD')


def extract(source, name):
    name_at = source.index('void ' + name + '(')
    start = source.rfind('template<', 0, name_at)
    brace = source.index('{', name_at)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


SHIM = r'''
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define __device__
#define __forceinline__ inline
struct uint4 { std::uint32_t x,y,z,w; };
struct float2 { float x,y; };
_Float16 __ushort_as_half(unsigned short x) { return std::bit_cast<_Float16>(x); }
float __half2float(_Float16 x) { return static_cast<float>(x); }
std::uint32_t NegateSelectedBytes(std::uint32_t x, std::uint32_t signs) {
  std::uint32_t out=0;
  for (int i=0;i<4;++i) {
    int v=(x>>(8*i))&255;
    if ((signs>>i)&1) v=-v;
    out|=(static_cast<std::uint32_t>(v)&255)<<(8*i);
  }
  return out;
}
int __builtin_amdgcn_sudot4(bool sa,std::uint32_t a,bool sb,std::uint32_t b,int sum,bool sat) {
  assert(sa&&sb&&!sat);
  for(int i=0;i<4;++i)
    sum+=static_cast<std::int8_t>(a>>(8*i))*static_cast<std::int8_t>(b>>(8*i));
  return sum;
}
'''

ORACLE = r'''
std::uint32_t rng=0x76b48;
std::uint32_t next() { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
std::uint32_t u32(const std::uint8_t* p) { std::uint32_t v; std::memcpy(&v,p,4);return v; }
float half_scale(const std::uint8_t* p) { unsigned short v;std::memcpy(&v,p,2);return __half2float(__ushort_as_half(v)); }
bool same(float a,float b) { return std::bit_cast<std::uint32_t>(a)==std::bit_cast<std::uint32_t>(b); }
template<bool S>
float reference(const std::uint8_t* bp,int half,const uint4* x,const float2* ds,
                int base,const std::uint32_t* grid,const std::uint8_t* signs,float sum) {
  const float d=half_scale(bp);
  for(int sub=0;sub<8;++sub) {
    const int global_sub=half*8+sub;
    const int sb32=global_sub/2;
    const auto aux=u32(bp+66+4*sb32);
    float scale;
    if constexpr(S) {
      int nib=(bp[106+sb32/2]>>(4*(sb32%2)))&15;
      scale=d*static_cast<float>(1+2*nib);
    } else scale=d*(0.5F+static_cast<float>(aux>>28))*0.5F;
    const uint4 qx=x[base+sub];
    const std::uint32_t words[4]={qx.x,qx.y,qx.z,qx.w};
    int dot=0;
    for(int i=0;i<16;++i) {
      int within=(global_sub%2)*16+i;
      int word=within/4;
      int ix=bp[2+8*sb32+word];
      if constexpr(S) ix|=((bp[66+sb32]>>word)&1)<<8;
      int mag=(grid[ix]>>(8*(within%4)))&255;
      int sg;
      if constexpr(S) sg=bp[74+sb32*4+within/8];
      else sg=signs[(aux>>(7*(within/8)))&127];
      if((sg>>(within%8))&1) mag=-mag;
      dot+=mag*static_cast<std::int8_t>(words[i/4]>>(8*(i%4)));
    }
    sum+=scale*(ds[base+sub].x*static_cast<float>(dot));
  }
  return sum;
}
std::uint64_t comparisons=0;
template<bool S,std::size_t B> void check() {
  constexpr int slots=40;
  std::uint32_t grid[512];std::uint8_t signs[128];
  for(unsigned i=0;i<512;++i)
    grid[i]=(1+(i&63))|((1+((i>>6)&7))<<8)|((1+((i*13)&63))<<16)|((1+((i*29)&63))<<24);
  for(unsigned i=0;i<128;++i) signs[i]=i|(__builtin_parity(i)<<7);
  for(int trial=0;trial<128;++trial) {
    // Extra first byte deliberately makes the packed GGUF block unaligned.
    std::array<std::uint8_t,(S?110:98)+1> bytes[4];
    const std::uint8_t* bpQ[4]={bytes[0].data()+1,bytes[1].data()+1,bytes[2].data()+1,bytes[3].data()+1};
    const std::uint8_t* bp[2]={bpQ[0],bpQ[1]};
    for(int r=0;r<4;++r) {
      for(auto &v:bytes[r]) v=next();
      unsigned short d=trial==0?0:static_cast<unsigned short>(0x3000+(next()&1023));
      if(trial%3==0) d|=0x8000;
      std::memcpy(bytes[r].data()+1,&d,2);
    }
    uint4 x[B*slots];float2 ds[B*slots];
    for(std::size_t i=0;i<B*slots;++i) { x[i]={next(),next(),next(),next()};ds[i]={std::ldexp(static_cast<float>(1+(next()&255)),-14),0}; }
    float paired[2][B],single[2][B],ref[2][B];
    float quad[4][B],child[4][B],cref[4][B];
    for(int r=0;r<2;++r) for(std::size_t t=0;t<B;++t)
      paired[r][t]=single[r][t]=ref[r][t]=static_cast<float>(int(t)-r)*0.125F;
    for(int r=0;r<4;++r) for(std::size_t t=0;t<B;++t)
      quad[r][t]=child[r][t]=cref[r][t]=static_cast<float>(int(t)-r)*0.125F;
    for(int half=0;half<2;++half) {
      const int base=(trial%4)*3;
      Iq3MultiRowHalfBlockDot<S,B,2>(bp,half,x,ds,base,slots,paired,grid,signs);
      for(int r=0;r<2;++r) {
        if constexpr(S) Iq3sHalfBlockDot<B>(bp[r],half,x,ds,base,slots,single[r],grid);
        else Iq3xxsHalfBlockDot<B>(bp[r],half,x,ds,base,slots,single[r],grid,signs);
        for(std::size_t t=0;t<B;++t) {
#ifndef IQ3_PAIR_R1_ONLY
          // This scalar oracle has a different contraction graph: only compare
          // its bits in off mode. Fast mode checks actual R2 against actual R1.
          ref[r][t]=reference<S>(bp[r],half,x+t*slots,ds+t*slots,base,grid,signs,ref[r][t]);
          const bool oracle_matches=same(paired[r][t],ref[r][t]);
#else
          const bool oracle_matches=true;
#endif
          if(!same(paired[r][t],single[r][t])||!oracle_matches) {
            std::fprintf(stderr,"FAIL S=%d B=%zu trial=%d half=%d row=%d token=%zu\n",S,B,trial,half,r,t);
            std::abort();
          }
          ++comparisons;
        }
      }
      // R4 (same four-row-body the kernel runs, guarded like the kernel):
      // every row against the same R1 helper and the same byte oracle.
      if constexpr (B >= 6) {
        Iq3MultiRowHalfBlockDot<S,B,4>(bpQ,half,x,ds,base,slots,quad,grid,signs);
        for(int r=0;r<4;++r) {
          if constexpr(S) Iq3sHalfBlockDot<B>(bpQ[r],half,x,ds,base,slots,child[r],grid);
          else Iq3xxsHalfBlockDot<B>(bpQ[r],half,x,ds,base,slots,child[r],grid,signs);
          for(std::size_t t=0;t<B;++t) {
#  ifndef IQ3_PAIR_R1_ONLY
            cref[r][t]=reference<S>(bpQ[r],half,x+t*slots,ds+t*slots,base,grid,signs,cref[r][t]);
            const bool oracle_matches4=same(quad[r][t],cref[r][t]);
#  else
            const bool oracle_matches4=true;
#  endif
            if(!same(quad[r][t],child[r][t])||!oracle_matches4) {
              std::fprintf(stderr,"FAIL4 S=%d B=%zu trial=%d half=%d row=%d token=%zu\n",S,B,trial,half,r,t);
              std::abort();
            }
            ++comparisons;
          }
        }
      }
    }
  }
}
int main() {
  check<true,4>();check<true,5>();check<true,6>();check<true,7>();check<true,8>();
  check<false,4>();check<false,5>();check<false,6>();check<false,7>();check<false,8>();
#ifdef IQ3_PAIR_R1_ONLY
  std::printf("PASS actual-helper vs R1 (FMA): %llu comparisons\n",(unsigned long long)comparisons);
#else
  std::printf("PASS actual-helper vs R1 + independent byte oracle: %llu comparisons\n",(unsigned long long)comparisons);
#endif
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    guard()
    source = HEADER.read_text()
    r1 = '\n'.join(extract(source,n) for n in ('Iq3sHalfBlockDot','Iq3xxsHalfBlockDot'))
    pair = extract(source, 'Iq3MultiRowHalfBlockDot')
    receipt = dict(header_sha256=hashlib.sha256(source.encode()).hexdigest(), tests=[])
    with tempfile.TemporaryDirectory() as directory:
        cpp,binary=Path(directory)/'check.cpp',Path(directory)/'check'
        for contract in ('off','fast'):
            guard();cpp.write_text(SHIM+r1+pair+ORACLE)
            flags=['-mfma']+(['-DIQ3_PAIR_R1_ONLY'] if contract=='fast' else [])
            subprocess.run([args.compiler,'-std=c++20','-O2',*flags,'-ffp-contract='+contract,str(cpp),'-o',str(binary)],check=True)
            assembly=subprocess.run(['objdump','-d','--no-show-raw-insn',str(binary)],check=True,text=True,capture_output=True).stdout
            fma_ops=re.findall(r'\bvfmadd\w+\b',assembly)
            if contract=='fast':
                assert fma_ops,'Fast-mode test emitted no FMA instructions'
            else:
                assert not fma_ops,'Off-mode test unexpectedly emitted FMA'
            result=subprocess.run([str(binary)],text=True,capture_output=True)
            assert result.returncode==0,result.stdout+result.stderr
            receipt['tests'].append(dict(fp_contract=contract,flags=flags,fma_instructions=len(fma_ops),fma_mnemonics=sorted(set(fma_ops)),oracle='R1 only' if contract=='fast' else 'R1 + independent byte oracle',result=result.stdout.strip()))
        # Execute a wrong-row mutant to prove distinct-row fixture sensitivity.
        guard();assert 'bp[r]' in pair
        cpp.write_text(SHIM+r1+pair.replace('bp[r]','bp[0]')+ORACLE)
        subprocess.run([args.compiler,'-std=c++20','-O2',str(cpp),'-o',str(binary)],check=True)
        result=subprocess.run([str(binary)],text=True,capture_output=True)
        assert result.returncode!=0,'Wrong-row mutation escaped fixture'
        receipt['negative_control']=result.stderr.strip()
    # Model unchanged grid/store ownership at awkward row tails; not GPU memory validation.
    for waves in (2,4,8):
        for rows in (2,4,8):
            for per_lane in (2,4):
                for m in (1,3,31,32,33,63,64,65):
                    seen=set()
                    for block in range((m+waves*rows*per_lane-1)//(waves*rows*per_lane)):
                        for thread in range(waves*32):
                            lanes=32//rows;unit=thread%32%lanes
                            first=((block*waves+thread//32)*rows+(thread%32)//lanes)*per_lane
                            for r in range(per_lane):
                                for t in range(8):
                                    if first+r<m and t%lanes==unit:
                                        item=(first+r,t);assert item not in seen;seen.add(item)
                    assert len(seen)==m*8
    receipt['tail_coordinate_models']=144
    receipt['limit']='Host executes actual helper source; does not qualify HIP codegen, LDS barriers, resources, native bit identity, quantization error/top1 or speed.'
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
