sudo apt update -y
apt install cuda-nsight-systems-12-4 -y
pip install uv -y
# export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

uv run nsys profile -o pytorch_profile  python benchmark.py
uv run nsys stats pytorch_profile.nsys-rep  > perf2.log