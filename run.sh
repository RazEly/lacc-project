#!/usr/bin/env bash
git clone https://www.github.com/RazEly/lacc-project
cd lacc-project
./init.sh
uv run python -m src.main
