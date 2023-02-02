
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
maxUploadSize = 300\n\
" > ~/.streamlit/config.toml