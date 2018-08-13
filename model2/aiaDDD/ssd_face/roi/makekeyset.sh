#!/bin/sh

s="[[:blank:]]"
w="[_[:alnum:]]"
c="[^[:blank:]#]"

sed=/bin/sed

if [ $# -lt 2 ] ; then
	echo "$0: usage FILE KEY [VALUE]"
	exit 1
fi

$sed -i -e "s/\(^$s*$2$s*:\?=$s*\)\($s*$c\+\)*\($s*#.*\)\?$/\1$3\3/" $1
