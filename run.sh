# make a new virtual env using python 3.11 (tested on MacOS 14.6.1)
# python3.11 -m venv .venv
# source .venv/bin/activate

# pip install -r requirements.txt

cd cleo

export UPSTAGE_API_KEY="upstage-api-key"
export TAVILY_API_KEY="tavily-api-key"

streamlit run service.py --server.port 8501 --server.address 0.0.0.0