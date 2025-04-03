echo "---------- Transfer from DDR ----------"
echo "Transfer only:"
python3 benchmark.py --gpu 
echo "Compute only:"
python3 benchmark.py --cpu
echo "Transfer and compute:"
python3 benchmark.py --gpu --cpu

echo "---------- Transfer from CXL ----------"
echo "Transfer only:"
python3 benchmark.py --gpu --cxl
echo "Compute only:"
python3 benchmark.py --cpu --cxl
echo "Transfer and compute:"
python3 benchmark.py --gpu --cpu --cxl