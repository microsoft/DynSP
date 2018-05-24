#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "Usage: run5fold.sh trialname"
	exit 9
fi

#alias mklpython='LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so python'
alias mklpython='LD_PRELOAD=/opt/intel/mkl/lib/libmkl_def.so:/opt/intel/mkl/lib/libmkl_avx2.so:/opt/intel/mkl/lib/libmkl_core.so:/opt/intel/mkl/lib/libmkl_intel_lp64.so:/opt/intel/mkl/lib/libmkl_intel_thread.so:/opt/intel/lib/libiomp5.so python2'
dm=2000
ds=1

for p in 0 1 2 3 4 5
do
	name="$1"-"$p"
	out=output/"$name".out
	model=model/"$name"
	log=log/"$name".log
	dirData=data
	echo $out
	mklpython sqafollow.py --dynet-mem $dm --dynet-seed $ds --expSym $p --model $model --log $log --dirData $dirData > $out 2>&1 &
	echo "Split: $p, PID: $!"
done
