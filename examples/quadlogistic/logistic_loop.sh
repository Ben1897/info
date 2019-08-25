# Loop over the tigramite_estimation_logistic.py w.r.t. different time lengths

# The time lengths
Trange=(200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
nnodes=55

for T in ${Trange[@]}
do
  mpirun -np $nnodes python tigramite_analysis_logistic.py $T
done