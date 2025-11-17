source miniforge/bin/activate
rm -rf ~/.local/share/pypoetry
rm -f ~/.local/bin/poetry
curl -sSL https://install.python-poetry.org | python -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
poetry config virtualenvs.in-project true --local
poetry env use "$(which python)"
poetry env info
poetry lock
poetry install