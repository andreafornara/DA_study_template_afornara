# source miniforge/bin/activate #source the python you want to use with poetry
# All the following commented comments are only needed if poetry is installed for the first time
# rm -rf ~/.local/share/pypoetry
# rm -f ~/.local/bin/poetry
# curl -sSL https://install.python-poetry.org | python -
# echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
# source ~/.bashrc
# The following commands are needed every time you want to create an env for a project
poetry config virtualenvs.in-project true --local
poetry env use "$(which python)"
poetry env info
poetry lock
poetry install