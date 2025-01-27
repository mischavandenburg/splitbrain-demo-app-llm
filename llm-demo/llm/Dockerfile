FROM ollama/ollama:latest

# Pre-Install llama2
RUN nohup bash -c "ollama serve &" && sleep 5 && ollama pull deepseek-r1:7b
# RUN nohup bash -c "ollama serve &" && wait4x http http://127.0.0.1:11434 && ollama pull deepseek-r1:7b
