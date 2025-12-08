
mpirun -n 1 ../../../../../mrhyde.exe input.yaml > mrhyde_np1.log
mpirun -n 2 ../../../../../mrhyde.exe input.yaml > mrhyde_np2.log
mpirun -n 4 ../../../../../mrhyde.exe input.yaml > mrhyde_np4.log
mpirun -n 8 ../../../../../mrhyde.exe input.yaml > mrhyde_np8.log
python strong_scaling.py mrhyde_np1.log mrhyde_np2.log mrhyde_np4.log mrhyde_np4.log --top-percent 20
