# Redis Vector Index Sizing Simulator


## Summary
Tool for generating VSS scenarios and then analyzing memory usage


## Features
- Calculates memory footprint of a Redis vector index scenario
## Prerequisites
- Python
- Docker
## Installation
1. Clone this repo.
```bash
git clone https://github.com/Redislabs-Solution-Architects/vss-sizer.git && cd vss-sizer
```
2. Install Python requirements (either in a virtual env or global)
```bash
pip install -r requirements.txt
```
3.  Start Redis Enterprise Docker environment
```bash
./start.sh
```

## Usage
### Options
- --url. URL connect string.  Default = redis://default:redis@localhost:12000
- --nkeys. Number of keys to be generated.  Default = 100,000.
- --indextype. Vector Index Type.  Default = flat.
- --metrictype.  Vector Metric Type.  Default = l2.
- --floattype.  Vector Float Type.  Default = f32.
- --vecdim.  Vector Dimension.  Default = 1536.
- --vecm.  HNSW M Param.  Default = 16.
### Execution
```bash
python3 vss-sizer.py --nkeys 100000 --objecttype hash --indextype flat --metrictype cosine 
--floattype f32 --vecdim 1536
```
### Output
Sample output for the test above.
```bash
Vector Index Test
 
*** Parameters ***
nkeys: 100000
objecttype: hash
indextype: flat
metrictype: cosine
floattype: float32
vecdim: 1536
 
*** Results ***
index ram used: 606.98 MB
data ram used: 812.24 MB
index to data ratio: 74.73%
document size: 7416 B
execution time: 4.37 sec
```
