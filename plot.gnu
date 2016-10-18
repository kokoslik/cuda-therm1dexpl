#!/usr/bin/gnuplot -p
unset key
plot for [i=0:5000:500] 'data.dat' index(i) w l
