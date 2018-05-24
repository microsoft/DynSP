#!/bin/sh

if [ "$#" -ne 1 ]; then
  echo "Usage: check.sh expSym"
  exit 1
fi

echo $1

awkscript='/ Accuracy/ { best[FILENAME] = $0; acc[FILENAME] = $3; iter[FILENAME] = $NF } 
END { n = asorti(best,sbest); 
	  for (i=1;i<=n;i++) { print best[sbest[i]] } 
	  for (i=2;i<=n;i++) { a += acc[sbest[i]]; s += iter[sbest[i]] } 
	  print a/5, int(s/5 + 0.5)
}'
gawk "$awkscript" output/"$1"-[0-5].out

