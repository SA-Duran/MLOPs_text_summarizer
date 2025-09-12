# MLOPs_text_summarizer
# MLOPs Text Summarizer

## Overview  
Text summarization tool that processes input texts via configurable pipelines, produces concise summaries, and provides an API for inference.

## Structure  

- `src/textSummarizer/`: core summarization logic and utilities  
- `research/`: notebooks and experiments for model exploration  
- `config/`: configuration files, e.g. `params.yaml`  
- `app.py` & `main.py`: API endpoints / script launches for inference  
- `template.py`: template utilities or wrapper logic  
- `Dockerfile`: containerization setup  
- `requirements.txt` / `pyproject.toml`: dependency management  
- `test.py`: basic tests or examples  

## Setup  

```bash
git clone https://github.com/SA-Duran/MLOPs_text_summarizer.git
cd MLOPs_text_summarizer
pip install -r requirements.txt
```

## Configuration  

Edit `config/params.yaml` to adjust parameters such as summary length, model type, etc.

## Running  

- To start the API / service:

  ```bash
  python app.py
  ```

- To run summarization from script:

  ```bash
  python main.py --input "Your input text here"
  ```

- To run tests / examples:

  ```bash
  python test.py
  ```

## Docker  

```bash
docker build -t text-summarizer .
docker run -p 5000:5000 text-summarizer
```

## Tools  

- Python  
- Flask or FastAPI for inference  
- Configuration via YAML  
- Docker  
- Jupyter notebooks  


## License  
MIT
