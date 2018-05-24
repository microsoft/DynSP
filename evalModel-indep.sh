#!/bin/sh

if [ "$#" -ne 2 ]; then
    echo "Usage: evalModel.sh trialname iter"
	exit 9
fi

#alias mklpython='LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so python'
alias mklpython='LD_PRELOAD=/opt/intel/mkl/lib/libmkl_def.so:/opt/intel/mkl/lib/libmkl_avx2.so:/opt/intel/mkl/lib/libmkl_core.so:/opt/intel/mkl/lib/libmkl_intel_lp64.so:/opt/intel/mkl/lib/libmkl_intel_thread.so:/opt/intel/lib/libiomp5.so python2'
dm=2000
ds=1

name="$1"-"0"
model=model/"$name"-"$2"
dirData=data
res=results/"$name"-"$2".tsv

mklpython sqafollow.py --dynet-mem $dm --dynet-seed $ds --expSym 0 --evalModel $model --dirData $dirData --res $res --indep
python eval.py data/test.tsv $res
